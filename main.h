//
//  main.h
//  LLaMAcpp
//
//  Created by Jed Fox on 2023-03-11.
//

//#pragma once
#include <sys/_types.h>
#include <vector>
#include <map>
#include <string>

#include <CoreFoundation/CFAvailability.h>

#include "utils.h"

struct foo {
    int i;
};

// default hparams (LLaMA 7B)
struct llama_hparams {
    int32_t n_vocab = 32000;
    int32_t n_ctx   = 512;   // this is provided as user input?
    int32_t n_embd  = 4096;
    int32_t n_mult  = 256;
    int32_t n_head  = 32;
    int32_t n_layer = 32;
    int32_t n_rot   = 64;
    int32_t f16     = 1;
};

struct llama_layer {
    // normalization
    struct ggml_tensor * attention_norm;

    // attention
    struct ggml_tensor * wq;
    struct ggml_tensor * wk;
    struct ggml_tensor * wv;
    struct ggml_tensor * wo;

    // normalization
    struct ggml_tensor * ffn_norm;

    // ff
    struct ggml_tensor * w1;
    struct ggml_tensor * w2;
    struct ggml_tensor * w3;
};

struct llama_model {
    llama_hparams hparams;

    struct ggml_tensor * tok_embeddings;

    struct ggml_tensor * norm;
    struct ggml_tensor * output;

    std::vector<llama_layer> layers;

    // key + value memory
    struct ggml_tensor * memory_k;
    struct ggml_tensor * memory_v;

    //
    struct ggml_context * ctx;
    std::map<std::string, struct ggml_tensor *> tensors;
};

struct llama_state {
    gpt_vocab vocab;
    llama_model model;
    struct {
        int64_t t_load_us = -1;
        int64_t t_sample_us = -1;
        int64_t t_predict_us = -1;
    } timing;
};

typedef CF_ENUM(unsigned int) {
    llama_stop_end_of_text,
    llama_stop_cancel,
    llama_stop_limit,
    llama_stop_error,
} llama_stop;

struct llama_progress {
    gpt_vocab::token token;
};

bool llama_bootstrap(const char *model_path, llama_state &state);
llama_stop llama_predict(gpt_params &params, llama_state &state, bool(^progress)(llama_progress));
void llama_finalize(llama_state &state);
