#pragma once

#include "ggml.h"

#include <cstdint>
#include <memory>

struct llama_model;

// Opt-in similarity-bounded cold-expert substitution for MoE models.
// Controlled by LLAMA_APPROX_ROUTING env var (disabled by default, zero overhead when off).
class llama_approx_routing {
public:
    static std::unique_ptr<llama_approx_routing> create_from_env(const llama_model & model);

    explicit llama_approx_routing(const llama_model & model);
    ~llama_approx_routing();

    bool is_layer_enabled(int32_t il, int64_t n_tokens) const;

    // Insert CPU custom op that rewrites cold low-weight expert ids using router similarity.
    // ids_in: contiguous I32 [n_expert_used, n_tokens]
    // weights: F32 [1, n_expert_used, n_tokens]
    ggml_tensor * build_rewrite(
            ggml_context * ctx0,
            ggml_tensor  * ids_in,
            ggml_tensor  * weights,
            int32_t        il) const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_approx_routing_ptr = std::unique_ptr<llama_approx_routing>;
