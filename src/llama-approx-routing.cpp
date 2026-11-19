#include "llama-approx-routing.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <numeric>
#include <string>
#include <vector>

#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {

enum class approx_mode {
    NONE = 0,
    TRACE,
    MINCORE,
    LRU,
    HOTSET,
};

enum class sub_policy {
    SIM = 0,
    RANDOM_RESIDENT,
    DROP,
};

struct sim_neighbor {
    int32_t id;
    float   sim;
};

struct exp_tensor_ref {
    const uint8_t * data = nullptr;
    size_t          stride = 0; // bytes per expert (nb[2])
};

struct approx_config {
    approx_mode mode          = approx_mode::NONE;
    sub_policy  policy        = sub_policy::SIM;
    float       tau           = 0.12f;
    float       delta         = 0.60f;
    int32_t     cap           = 0;     // 0 -> derive from cap_pct
    int32_t     cap_pct       = 50;
    int32_t     protect_top   = 1;     // never replace top-N highest-weight experts per token
    int32_t     top_neighbors = 32;
    int32_t     mincore_pct   = 75;
    int32_t     min_tokens    = 0;
    int32_t     max_tokens    = 0;
    uint64_t    warmup_tokens = 512;
    bool        centered      = true;
    bool        print_stats   = true;
    std::string trace_file;
    std::string hotset_file;
};

static const char * mode_name(approx_mode m) {
    switch (m) {
        case approx_mode::TRACE:   return "trace";
        case approx_mode::MINCORE: return "mincore";
        case approx_mode::LRU:     return "lru";
        case approx_mode::HOTSET:  return "hotset";
        default:                   return "none";
    }
}

static bool parse_env_config(approx_config & cfg) {
    const char * env_mode = std::getenv("LLAMA_APPROX_ROUTING");
    if (!env_mode || !env_mode[0] || std::strcmp(env_mode, "0") == 0 || std::strcmp(env_mode, "off") == 0) {
        return false;
    }

    if (std::strcmp(env_mode, "1") == 0 || std::strcmp(env_mode, "lru") == 0) {
        cfg.mode = approx_mode::LRU;
    } else if (std::strcmp(env_mode, "mincore") == 0) {
        cfg.mode = approx_mode::MINCORE;
    } else if (std::strcmp(env_mode, "hotset") == 0) {
        cfg.mode = approx_mode::HOTSET;
    } else if (std::strcmp(env_mode, "trace") == 0) {
        cfg.mode = approx_mode::TRACE;
    } else {
        LLAMA_LOG_WARN("llama_approx_routing: unknown LLAMA_APPROX_ROUTING='%s', disabling\n", env_mode);
        return false;
    }

    if (const char * v = std::getenv("LLAMA_APPROX_SUB_POLICY")) {
        if (std::strcmp(v, "random") == 0 || std::strcmp(v, "random_resident") == 0) {
            cfg.policy = sub_policy::RANDOM_RESIDENT;
        } else if (std::strcmp(v, "drop") == 0) {
            cfg.policy = sub_policy::DROP;
        }
    }
    if (const char * v = std::getenv("LLAMA_APPROX_CENTERED")) {
        cfg.centered = std::atoi(v) != 0;
    }
    if (const char * v = std::getenv("LLAMA_APPROX_TAU")) {
        cfg.tau = std::strtof(v, nullptr);
    }
    if (const char * v = std::getenv("LLAMA_APPROX_DELTA")) {
        cfg.delta = std::strtof(v, nullptr);
    }
    if (const char * v = std::getenv("LLAMA_APPROX_CAP")) {
        cfg.cap = std::atoi(v);
    }
    if (const char * v = std::getenv("LLAMA_APPROX_CAP_PCT")) {
        cfg.cap_pct = std::clamp(std::atoi(v), 1, 100);
    }
    if (const char * v = std::getenv("LLAMA_APPROX_PROTECT_TOP")) {
        cfg.protect_top = std::max(0, std::atoi(v));
    }
    if (const char * v = std::getenv("LLAMA_APPROX_TOP_NEIGHBORS")) {
        cfg.top_neighbors = std::clamp(std::atoi(v), 1, 256);
    }
    if (const char * v = std::getenv("LLAMA_APPROX_MINCORE_PCT")) {
        cfg.mincore_pct = std::clamp(std::atoi(v), 1, 100);
    }
    if (const char * v = std::getenv("LLAMA_APPROX_MIN_TOKENS")) {
        cfg.min_tokens = std::max(0, std::atoi(v));
    }
    if (const char * v = std::getenv("LLAMA_APPROX_MAX_TOKENS")) {
        cfg.max_tokens = std::max(0, std::atoi(v));
    }
    if (const char * v = std::getenv("LLAMA_APPROX_WARMUP_TOKENS")) {
        cfg.warmup_tokens = (uint64_t) std::max(0LL, std::atoll(v));
    }
    if (const char * v = std::getenv("LLAMA_APPROX_STATS")) {
        cfg.print_stats = std::atoi(v) != 0;
    }
    if (const char * v = std::getenv("LLAMA_APPROX_TRACE_FILE")) {
        cfg.trace_file = v;
    } else if (cfg.mode == approx_mode::TRACE) {
        cfg.trace_file = "/tmp/moe_trace.bin";
    }
    if (const char * v = std::getenv("LLAMA_APPROX_HOTSET_FILE")) {
        cfg.hotset_file = v;
    }

    return true;
}

static void convert_row_to_f32(const ggml_tensor * t, const std::vector<uint8_t> & raw, int64_t row, float * out) {
    const int64_t n = t->ne[0];
    const uint8_t * src = raw.data() + (size_t) row * t->nb[1];
    switch (t->type) {
        case GGML_TYPE_F32:
            std::memcpy(out, src, (size_t) n * sizeof(float));
            break;
        case GGML_TYPE_F16: {
            const ggml_fp16_t * p = reinterpret_cast<const ggml_fp16_t *>(src);
            for (int64_t i = 0; i < n; ++i) {
                out[i] = ggml_fp16_to_fp32(p[i]);
            }
        } break;
        case GGML_TYPE_BF16: {
            const ggml_bf16_t * p = reinterpret_cast<const ggml_bf16_t *>(src);
            for (int64_t i = 0; i < n; ++i) {
                out[i] = ggml_bf16_to_fp32(p[i]);
            }
        } break;
        default:
            std::memset(out, 0, (size_t) n * sizeof(float));
            break;
    }
}

struct layer_state {
    int32_t il       = -1;
    int32_t n_expert = 0;
    int32_t cap      = 0;
    int32_t k_nbr    = 0;
    bool    enabled  = false;

    // Flattened [n_expert * k_nbr] sorted descending by cosine similarity
    std::vector<sim_neighbor> sim_table;

    // Host mmap expert weight references for mincore mode
    std::vector<exp_tensor_ref> exp_tensors;

    // LRU state
    std::vector<uint8_t>  lru_present;
    std::vector<uint64_t> lru_last_tick;
    int32_t               lru_count = 0;
    uint64_t              lru_tick  = 0;

    // Hotset state
    std::vector<uint64_t> hot_freq;
    std::vector<uint8_t>  hot_present;
    uint64_t              hot_tokens = 0;
    bool                  hot_frozen = false;

    // Scratch buffers reused across calls (protected by layer execution being single-threaded)
    std::vector<int8_t>  batch_mincore; // -1 unknown, 0 cold, 1 resident
    std::vector<uint8_t> mincore_vec;

    // RNG state for random_resident ablation
    uint64_t rng_state = 0x9e3779b97f4a7c15ULL;

    // Telemetry
    uint64_t total_tokens       = 0;
    uint64_t total_selections   = 0;
    uint64_t resident_hits      = 0;
    uint64_t cold_misses        = 0;
    uint64_t cold_substituted   = 0;
    uint64_t cold_kept_tau      = 0;
    uint64_t cold_kept_protect  = 0;
    uint64_t cold_kept_nosim    = 0;
    uint64_t sub_dup_hits       = 0;
    double   sub_sim_sum        = 0.0;
    double   sub_w_sum          = 0.0;

    uint32_t next_rand() {
        uint64_t x = rng_state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        rng_state = x;
        return (uint32_t) ((x * 0x2545F4914F6CDD1DULL) >> 32);
    }
};

struct layer_cb_ctx {
    const approx_config * cfg       = nullptr;
    layer_state         * st        = nullptr;
    size_t                page_size = 4096;
    FILE                * trace_fp  = nullptr;
    std::mutex          * trace_mtx = nullptr;
};

static bool check_expert_mincore(layer_state & st, int32_t e, size_t page_size, int32_t mincore_pct) {
    if (st.batch_mincore[e] >= 0) {
        return st.batch_mincore[e] != 0;
    }
    if (st.exp_tensors.empty()) {
        st.batch_mincore[e] = 1;
        return true;
    }
#if defined(__linux__)
    size_t total_pages = 0;
    size_t res_pages   = 0;
    for (const auto & ref : st.exp_tensors) {
        const uintptr_t start = reinterpret_cast<uintptr_t>(ref.data + (size_t) e * ref.stride);
        const uintptr_t end   = start + ref.stride;
        const uintptr_t p0    = start & ~(uintptr_t)(page_size - 1);
        const uintptr_t p1    = (end + page_size - 1) & ~(uintptr_t)(page_size - 1);
        const size_t    len   = (size_t) (p1 - p0);
        const size_t    np    = len / page_size;
        if (np == 0) {
            continue;
        }
        if (st.mincore_vec.size() < np) {
            st.mincore_vec.resize(np);
        }
        if (mincore(reinterpret_cast<void *>(p0), len, st.mincore_vec.data()) != 0) {
            // Non-mmap or unsupported buffer: treat as resident
            st.batch_mincore[e] = 1;
            return true;
        }
        total_pages += np;
        for (size_t i = 0; i < np; ++i) {
            res_pages += (st.mincore_vec[i] & 1u);
        }
    }
    const bool is_res = (total_pages == 0) || (res_pages * 100 >= total_pages * (size_t) mincore_pct);
    st.batch_mincore[e] = is_res ? 1 : 0;
    return is_res;
#else
    (void) page_size;
    (void) mincore_pct;
    st.batch_mincore[e] = 1;
    return true;
#endif
}

static void lru_touch(layer_state & st, int32_t e) {
    st.lru_tick++;
    if (st.lru_present[e]) {
        st.lru_last_tick[e] = st.lru_tick;
        return;
    }
    if (st.lru_count >= st.cap) {
        int32_t victim = -1;
        uint64_t best_t = UINT64_MAX;
        for (int32_t i = 0; i < st.n_expert; ++i) {
            if (st.lru_present[i] && st.lru_last_tick[i] < best_t) {
                best_t = st.lru_last_tick[i];
                victim = i;
            }
        }
        if (victim >= 0) {
            st.lru_present[victim] = 0;
            st.lru_count--;
        }
    }
    st.lru_present[e]   = 1;
    st.lru_last_tick[e] = st.lru_tick;
    st.lru_count++;
}

static void freeze_hotset(layer_state & st) {
    std::vector<int32_t> order(st.n_expert);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
        if (st.hot_freq[a] != st.hot_freq[b]) {
            return st.hot_freq[a] > st.hot_freq[b];
        }
        return a < b;
    });
    std::fill(st.hot_present.begin(), st.hot_present.end(), 0);
    const int32_t keep = std::min(st.cap, st.n_expert);
    for (int32_t i = 0; i < keep; ++i) {
        st.hot_present[order[i]] = 1;
    }
    st.hot_frozen = true;
}

static void approx_routing_custom_op(
        ggml_tensor       * dst,
        const ggml_tensor * a, // selected_experts [n_expert_used, n_tokens] I32
        const ggml_tensor * b, // weights          [1, n_expert_used, n_tokens] F32
        int                 ith,
        int                 nth,
        void              * userdata) {
    (void) nth;
    if (ith != 0) {
        return;
    }

    auto * cb = static_cast<layer_cb_ctx *>(userdata);
    const approx_config & cfg = *cb->cfg;
    layer_state & st = *cb->st;

    const int64_t n_expert_used = a->ne[0];
    const int64_t n_tokens      = a->ne[1];

    if (cfg.mode == approx_mode::MINCORE) {
        std::fill(st.batch_mincore.begin(), st.batch_mincore.end(), (int8_t) -1);
    }

    std::array<int32_t, 64> orig_ids{};
    std::array<int32_t, 64> out_ids{};
    std::array<float,   64> w_rel{};
    std::array<int32_t, 64> rank_order{};

    const int64_t k = std::min<int64_t>(n_expert_used, 64);

    // Optional trace buffer for this batch
    std::vector<uint8_t> trace_buf;
    if (cb->trace_fp != nullptr) {
        trace_buf.reserve((size_t) n_tokens * (4 + (size_t) k * 8));
    }

    for (int64_t t = 0; t < n_tokens; ++t) {
        const char * a_row   = (const char *) a->data + (size_t) t * a->nb[1];
        const char * b_tok   = (const char *) b->data + (size_t) t * b->nb[2];
        char       * dst_row = (char *)       dst->data + (size_t) t * dst->nb[1];

        float w_sum = 0.0f;
        for (int64_t i = 0; i < k; ++i) {
            orig_ids[i] = *reinterpret_cast<const int32_t *>(a_row + (size_t) i * a->nb[0]);
            out_ids[i]  = orig_ids[i];
            const float wi = *reinterpret_cast<const float *>(b_tok + (size_t) i * b->nb[1]);
            w_rel[i] = wi;
            w_sum += std::fabs(wi);
            rank_order[i] = (int32_t) i;
        }
        // Copy any extra slots beyond 64 untouched
        for (int64_t i = k; i < n_expert_used; ++i) {
            const int32_t id = *reinterpret_cast<const int32_t *>(a_row + (size_t) i * a->nb[0]);
            *reinterpret_cast<int32_t *>(dst_row + (size_t) i * dst->nb[0]) = id;
        }

        if (w_sum > 1e-12f) {
            const float inv = 1.0f / w_sum;
            for (int64_t i = 0; i < k; ++i) {
                w_rel[i] *= inv;
            }
        } else if (k > 0) {
            const float inv = 1.0f / (float) k;
            for (int64_t i = 0; i < k; ++i) {
                w_rel[i] = inv;
            }
        }

        if (cb->trace_fp != nullptr) {
            const uint16_t il16 = (uint16_t) st.il;
            const uint16_t k16  = (uint16_t) k;
            const size_t pos = trace_buf.size();
            trace_buf.resize(pos + 4 + (size_t) k * (sizeof(int32_t) + sizeof(float)));
            std::memcpy(trace_buf.data() + pos,     &il16, 2);
            std::memcpy(trace_buf.data() + pos + 2, &k16,  2);
            std::memcpy(trace_buf.data() + pos + 4, orig_ids.data(), (size_t) k * sizeof(int32_t));
            std::memcpy(trace_buf.data() + pos + 4 + (size_t) k * sizeof(int32_t), w_rel.data(), (size_t) k * sizeof(float));
        }

        st.total_tokens++;
        st.total_selections += (uint64_t) k;

        if (cfg.mode == approx_mode::TRACE) {
            for (int64_t i = 0; i < k; ++i) {
                *reinterpret_cast<int32_t *>(dst_row + (size_t) i * dst->nb[0]) = orig_ids[i];
            }
            continue;
        }

        // Hotset warmup phase
        if (cfg.mode == approx_mode::HOTSET && !st.hot_frozen) {
            for (int64_t i = 0; i < k; ++i) {
                if (orig_ids[i] >= 0 && orig_ids[i] < st.n_expert) {
                    st.hot_freq[orig_ids[i]]++;
                }
                *reinterpret_cast<int32_t *>(dst_row + (size_t) i * dst->nb[0]) = orig_ids[i];
            }
            st.hot_tokens++;
            st.resident_hits += (uint64_t) k;
            if (st.hot_tokens >= cfg.warmup_tokens) {
                freeze_hotset(st);
            }
            continue;
        }

        // Sort slots by relative weight descending so highest-weight experts are protected and resolved first
        std::sort(rank_order.begin(), rank_order.begin() + k, [&](int32_t ia, int32_t ib) {
            return w_rel[ia] > w_rel[ib];
        });

        auto is_resident = [&](int32_t e) -> bool {
            if (e < 0 || e >= st.n_expert) {
                return true;
            }
            switch (cfg.mode) {
                case approx_mode::MINCORE:
                    return check_expert_mincore(st, e, cb->page_size, cfg.mincore_pct);
                case approx_mode::LRU:
                    return st.lru_present[e] != 0 || st.lru_count < st.cap;
                case approx_mode::HOTSET:
                    return st.hot_present[e] != 0;
                default:
                    return true;
            }
        };

        auto mark_used = [&](int32_t e) {
            if (e < 0 || e >= st.n_expert) {
                return;
            }
            if (cfg.mode == approx_mode::LRU) {
                lru_touch(st, e);
            } else if (cfg.mode == approx_mode::MINCORE) {
                st.batch_mincore[e] = 1;
            }
        };

        for (int64_t r = 0; r < k; ++r) {
            const int32_t slot = rank_order[r];
            const int32_t e    = orig_ids[slot];
            if (e < 0 || e >= st.n_expert) {
                continue;
            }

            if (is_resident(e)) {
                st.resident_hits++;
                mark_used(e);
                continue;
            }

            // Expert is cold
            st.cold_misses++;

            if (r < cfg.protect_top) {
                st.cold_kept_protect++;
                mark_used(e);
                continue;
            }

            if (w_rel[slot] >= cfg.tau) {
                st.cold_kept_tau++;
                mark_used(e);
                continue;
            }

            auto already_in_token = [&](int32_t cand) -> bool {
                for (int64_t j = 0; j < k; ++j) {
                    if (out_ids[j] == cand) {
                        return true;
                    }
                }
                return false;
            };

            int32_t best_sub = -1;
            float   best_sim = -1.0f;

            if (cfg.policy == sub_policy::DROP) {
                // Skip cold low-weight expert on CPU (MUL_MAT_ID handles id == -1 by zeroing output slot)
                out_ids[slot] = -1;
                st.cold_substituted++;
                st.sub_w_sum += (double) w_rel[slot];
                continue;
            } else if (cfg.policy == sub_policy::SIM) {
                const sim_neighbor * nbrs = st.sim_table.data() + (size_t) e * (size_t) st.k_nbr;
                for (int32_t j = 0; j < st.k_nbr; ++j) {
                    if (nbrs[j].sim < cfg.delta) {
                        break;
                    }
                    const int32_t cand = nbrs[j].id;
                    if (cand >= 0 && !already_in_token(cand) && is_resident(cand)) {
                        best_sub = cand;
                        best_sim = nbrs[j].sim;
                        break;
                    }
                }
            } else {
                // Random resident ablation control: pick a random distinct resident expert if any neighbor meets delta
                bool has_sim_eligible = false;
                const sim_neighbor * nbrs = st.sim_table.data() + (size_t) e * (size_t) st.k_nbr;
                for (int32_t j = 0; j < st.k_nbr; ++j) {
                    if (nbrs[j].sim < cfg.delta) {
                        break;
                    }
                    if (nbrs[j].id >= 0 && !already_in_token(nbrs[j].id) && is_resident(nbrs[j].id)) {
                        has_sim_eligible = true;
                        break;
                    }
                }
                if (has_sim_eligible) {
                    const uint32_t start = st.next_rand() % (uint32_t) st.n_expert;
                    for (int32_t off = 0; off < st.n_expert; ++off) {
                        const int32_t cand = (int32_t) ((start + (uint32_t) off) % (uint32_t) st.n_expert);
                        if (cand != e && !already_in_token(cand) && is_resident(cand)) {
                            best_sub = cand;
                            best_sim = 0.0f;
                            break;
                        }
                    }
                }
            }

            if (best_sub >= 0) {
                bool dup = false;
                for (int64_t j = 0; j < k; ++j) {
                    if (j != slot && out_ids[j] == best_sub) {
                        dup = true;
                        break;
                    }
                }
                if (dup) {
                    st.sub_dup_hits++;
                }
                out_ids[slot] = best_sub;
                st.cold_substituted++;
                st.sub_sim_sum += (double) best_sim;
                st.sub_w_sum   += (double) w_rel[slot];
                mark_used(best_sub);
            } else {
                st.cold_kept_nosim++;
                mark_used(e);
            }
        }

        for (int64_t i = 0; i < k; ++i) {
            *reinterpret_cast<int32_t *>(dst_row + (size_t) i * dst->nb[0]) = out_ids[i];
        }
    }

    if (cb->trace_fp != nullptr && !trace_buf.empty()) {
        std::lock_guard<std::mutex> lock(*cb->trace_mtx);
        std::fwrite(trace_buf.data(), 1, trace_buf.size(), cb->trace_fp);
    }
}

} // namespace

struct llama_approx_routing::impl {
    approx_config             cfg;
    size_t                    page_size = 4096;
    std::vector<layer_state>  layers;
    std::vector<layer_cb_ctx> cb_ctxs; // stable pointers after init
    FILE                    * trace_fp = nullptr;
    std::mutex                trace_mtx;

    explicit impl(const llama_model & model, const approx_config & c) : cfg(c) {
#if defined(__linux__)
        const long ps = sysconf(_SC_PAGESIZE);
        if (ps > 0) {
            page_size = (size_t) ps;
        }
#endif
        const int32_t n_layer  = (int32_t) model.layers.size();
        const int32_t n_expert = (int32_t) model.hparams.n_expert;

        layers.resize(n_layer);
        cb_ctxs.resize(n_layer);

        int32_t active_layers = 0;
        int32_t host_exp_layers = 0;
        int32_t sample_cap = cfg.cap;

        for (int32_t il = 0; il < n_layer; ++il) {
            auto & st = layers[il];
            st.il = il;

            const auto & ml = model.layers[il];
            ggml_tensor * gate_inp = ml.ffn_gate_inp;
            if (!gate_inp || gate_inp->data == nullptr) {
                continue;
            }
            const int32_t layer_n_exp = (int32_t) gate_inp->ne[1];
            const int32_t n_embd      = (int32_t) gate_inp->ne[0];
            if (layer_n_exp < 2 || n_embd < 1) {
                continue;
            }

            st.n_expert = layer_n_exp;
            st.cap      = cfg.cap > 0 ? std::min(cfg.cap, layer_n_exp) : std::max(1, (layer_n_exp * cfg.cap_pct) / 100);
            st.k_nbr    = std::min(cfg.top_neighbors, layer_n_exp - 1);
            st.enabled  = true;
            st.rng_state ^= (uint64_t) (il + 1) * 0xbf58476d1ce4e5b9ULL;

            // Collect host-resident expert tensors for mincore residency checks
            for (ggml_tensor * w : { ml.ffn_gate_exps, ml.ffn_up_exps, ml.ffn_down_exps, ml.ffn_gate_up_exps }) {
                if (w != nullptr && w->data != nullptr && w->buffer != nullptr && ggml_backend_buffer_is_host(w->buffer)) {
                    st.exp_tensors.push_back({ static_cast<const uint8_t *>(w->data), w->nb[2] });
                }
            }
            if (!st.exp_tensors.empty()) {
                host_exp_layers++;
            }

            st.lru_present.assign(layer_n_exp, 0);
            st.lru_last_tick.assign(layer_n_exp, 0);
            st.hot_freq.assign(layer_n_exp, 0);
            st.hot_present.assign(layer_n_exp, 0);
            st.batch_mincore.assign(layer_n_exp, -1);

            // Build top-k cosine similarity table from router weight rows (skip in pure trace mode)
            if (cfg.mode != approx_mode::TRACE) {
                const size_t nbytes = ggml_nbytes(gate_inp);
                std::vector<uint8_t> raw(nbytes);
                ggml_backend_tensor_get(gate_inp, raw.data(), 0, nbytes);

                std::vector<float> rows((size_t) layer_n_exp * (size_t) n_embd);
                for (int32_t e = 0; e < layer_n_exp; ++e) {
                    float * row = rows.data() + (size_t) e * (size_t) n_embd;
                    convert_row_to_f32(gate_inp, raw, e, row);
                }
                if (cfg.centered) {
                    std::vector<float> mu(n_embd, 0.0f);
                    for (int32_t e = 0; e < layer_n_exp; ++e) {
                        const float * row = rows.data() + (size_t) e * (size_t) n_embd;
                        for (int32_t d = 0; d < n_embd; ++d) {
                            mu[d] += row[d];
                        }
                    }
                    const float inv_e = 1.0f / (float) layer_n_exp;
                    for (int32_t d = 0; d < n_embd; ++d) {
                        mu[d] *= inv_e;
                    }
                    for (int32_t e = 0; e < layer_n_exp; ++e) {
                        float * row = rows.data() + (size_t) e * (size_t) n_embd;
                        for (int32_t d = 0; d < n_embd; ++d) {
                            row[d] -= mu[d];
                        }
                    }
                }
                for (int32_t e = 0; e < layer_n_exp; ++e) {
                    float * row = rows.data() + (size_t) e * (size_t) n_embd;
                    double norm2 = 0.0;
                    for (int32_t d = 0; d < n_embd; ++d) {
                        norm2 += (double) row[d] * (double) row[d];
                    }
                    const float inv_norm = norm2 > 1e-24 ? (float) (1.0 / std::sqrt(norm2)) : 0.0f;
                    for (int32_t d = 0; d < n_embd; ++d) {
                        row[d] *= inv_norm;
                    }
                }

                st.sim_table.resize((size_t) layer_n_exp * (size_t) st.k_nbr);
                std::vector<sim_neighbor> tmp(layer_n_exp - 1);
                for (int32_t e = 0; e < layer_n_exp; ++e) {
                    const float * r_e = rows.data() + (size_t) e * (size_t) n_embd;
                    int32_t idx = 0;
                    for (int32_t j = 0; j < layer_n_exp; ++j) {
                        if (j == e) {
                            continue;
                        }
                        const float * r_j = rows.data() + (size_t) j * (size_t) n_embd;
                        float dot = 0.0f;
                        for (int32_t d = 0; d < n_embd; ++d) {
                            dot += r_e[d] * r_j[d];
                        }
                        tmp[idx++] = { j, dot };
                    }
                    std::partial_sort(tmp.begin(), tmp.begin() + st.k_nbr, tmp.end(),
                        [](const sim_neighbor & x, const sim_neighbor & y) { return x.sim > y.sim; });
                    std::memcpy(st.sim_table.data() + (size_t) e * (size_t) st.k_nbr,
                                tmp.data(), (size_t) st.k_nbr * sizeof(sim_neighbor));
                }
            }

            if (active_layers == 0) {
                sample_cap = st.cap;
            }
            active_layers++;
        }

        // Load static hotset file if provided: binary { magic=0x484F5453, n_layer, n_expert, uint8_t mask[n_layer][n_expert] }
        if (cfg.mode == approx_mode::HOTSET && !cfg.hotset_file.empty()) {
            std::ifstream f(cfg.hotset_file, std::ios::binary);
            uint32_t hdr[3] = { 0, 0, 0 };
            if (f && f.read(reinterpret_cast<char *>(hdr), sizeof(hdr)) && hdr[0] == 0x484F5453u) {
                const int32_t fl = (int32_t) hdr[1];
                const int32_t fe = (int32_t) hdr[2];
                for (int32_t il = 0; il < fl && il < n_layer; ++il) {
                    std::vector<uint8_t> mask(fe, 0);
                    f.read(reinterpret_cast<char *>(mask.data()), fe);
                    if (layers[il].enabled && layers[il].n_expert == fe) {
                        layers[il].hot_present = std::move(mask);
                        layers[il].hot_frozen  = true;
                    }
                }
                LLAMA_LOG_INFO("llama_approx_routing: loaded hotset from '%s'\n", cfg.hotset_file.c_str());
            } else {
                LLAMA_LOG_WARN("llama_approx_routing: failed to load hotset '%s', falling back to warmup\n", cfg.hotset_file.c_str());
            }
        }

        if (!cfg.trace_file.empty()) {
            trace_fp = std::fopen(cfg.trace_file.c_str(), "wb");
            if (trace_fp) {
                const uint32_t hdr[4] = {
                    0x54524143u, // "TRAC"
                    (uint32_t) n_layer,
                    (uint32_t) n_expert,
                    (uint32_t) model.hparams.n_expert_used_max(),
                };
                std::fwrite(hdr, sizeof(uint32_t), 4, trace_fp);
                LLAMA_LOG_INFO("llama_approx_routing: tracing routing decisions to '%s'\n", cfg.trace_file.c_str());
            } else {
                LLAMA_LOG_WARN("llama_approx_routing: could not open trace file '%s'\n", cfg.trace_file.c_str());
            }
        }

        for (int32_t il = 0; il < n_layer; ++il) {
            cb_ctxs[il] = { &cfg, &layers[il], page_size, trace_fp, &trace_mtx };
        }

        const char * pol_str = cfg.policy == sub_policy::SIM ? "sim" : (cfg.policy == sub_policy::DROP ? "drop" : "random_resident");
        std::fprintf(stderr, "[APPROX_ROUTING_INIT] mode=%s policy=%s centered=%d layers=%d (host_exp=%d) tau=%.3f delta=%.3f cap=%d (%d%%) protect_top=%d\n",
                mode_name(cfg.mode),
                pol_str,
                (int) cfg.centered,
                active_layers, host_exp_layers,
                cfg.tau, cfg.delta,
                sample_cap,
                cfg.cap_pct, cfg.protect_top);
        std::fflush(stderr);
    }

    ~impl() {
        if (trace_fp) {
            std::fclose(trace_fp);
            trace_fp = nullptr;
        }
        if (!cfg.print_stats) {
            return;
        }
        uint64_t tot_sel = 0, hits = 0, misses = 0, subs = 0, kept_tau = 0, kept_prot = 0, kept_nosim = 0, dups = 0;
        double sim_sum = 0.0, w_sum = 0.0;
        for (const auto & st : layers) {
            if (!st.enabled) {
                continue;
            }
            tot_sel    += st.total_selections;
            hits       += st.resident_hits;
            misses     += st.cold_misses;
            subs       += st.cold_substituted;
            kept_tau   += st.cold_kept_tau;
            kept_prot  += st.cold_kept_protect;
            kept_nosim += st.cold_kept_nosim;
            dups       += st.sub_dup_hits;
            sim_sum    += st.sub_sim_sum;
            w_sum      += st.sub_w_sum;
        }
        if (tot_sel == 0) {
            return;
        }
        const double hit_pct    = 100.0 * (double) hits / (double) tot_sel;
        const double miss_pct   = 100.0 * (double) misses / (double) tot_sel;
        const double sub_tot    = 100.0 * (double) subs / (double) tot_sel;
        const double sub_of_m   = misses > 0 ? 100.0 * (double) subs / (double) misses : 0.0;
        const double mean_sim   = subs > 0 ? sim_sum / (double) subs : 0.0;
        const double mean_w     = subs > 0 ? w_sum / (double) subs : 0.0;
        const uint64_t rem_faults = misses - subs;

        const char * pol_str = cfg.policy == sub_policy::SIM ? "sim" : (cfg.policy == sub_policy::DROP ? "drop" : "random_resident");
        std::fprintf(stderr, "[APPROX_ROUTING_STATS] mode=%s policy=%s centered=%d tau=%.3f delta=%.3f total_sel=%llu hit=%llu (%.2f%%) cold_miss=%llu (%.2f%%) sub=%llu (%.2f%% tot, %.2f%% of misses) rem_faults=%llu kept(prot=%llu tau=%llu nosim=%llu) dup_sub=%llu mean_sim=%.4f mean_w=%.4f\n",
                mode_name(cfg.mode),
                pol_str,
                (int) cfg.centered,
                cfg.tau, cfg.delta,
                (unsigned long long) tot_sel,
                (unsigned long long) hits, hit_pct,
                (unsigned long long) misses, miss_pct,
                (unsigned long long) subs, sub_tot, sub_of_m,
                (unsigned long long) rem_faults,
                (unsigned long long) kept_prot,
                (unsigned long long) kept_tau,
                (unsigned long long) kept_nosim,
                (unsigned long long) dups,
                mean_sim, mean_w);
        std::fflush(stderr);
    }
};

std::unique_ptr<llama_approx_routing> llama_approx_routing::create_from_env(const llama_model & model) {
    approx_config cfg;
    if (!parse_env_config(cfg)) {
        return nullptr;
    }
    return std::unique_ptr<llama_approx_routing>(new llama_approx_routing(model));
}

llama_approx_routing::llama_approx_routing(const llama_model & model) {
    approx_config cfg;
    parse_env_config(cfg);
    pimpl = std::make_unique<impl>(model, cfg);
}

llama_approx_routing::~llama_approx_routing() = default;

bool llama_approx_routing::is_layer_enabled(int32_t il, int64_t n_tokens) const {
    if (!pimpl || il < 0 || il >= (int32_t) pimpl->layers.size()) {
        return false;
    }
    if (!pimpl->layers[il].enabled) {
        return false;
    }
    if (pimpl->cfg.min_tokens > 0 && n_tokens < pimpl->cfg.min_tokens) {
        return false;
    }
    if (pimpl->cfg.max_tokens > 0 && n_tokens > pimpl->cfg.max_tokens) {
        return false;
    }
    return true;
}

ggml_tensor * llama_approx_routing::build_rewrite(
        ggml_context * ctx0,
        ggml_tensor  * ids_in,
        ggml_tensor  * weights,
        int32_t        il) const {
    return ggml_map_custom2(
            ctx0,
            ids_in,
            weights,
            approx_routing_custom_op,
            1,
            &pimpl->cb_ctxs[il]);
}
