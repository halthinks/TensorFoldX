# Qwen3 text checkpoints on CUDA (Qwen3ForCausalLM, BF16)

A Qwen3 folder that has no sentence-transformers head (`1_Pooling/config.json` or `1_LogitScore/config.json`) is a
text model. On CUDA it opens the shared dense engine (`zig/src/cuda_dense`) with the Qwen3 recipe in
`zig/src/families/qwen3/text_recipe.zig`; a folder with such a head opens the embedding and reranker engine as before.

## Contract

- Added: text generation for `model_type` `qwen3` in BF16, head dim 128, no attention bias, plain RoPE, tied or untied
  output head. Other configs are refused by name (`config.zig`, `text_recipe.zig`'s `explain`).
- Checkpoint used: `wyt2000/CodeV-SVA-8B` at `56b9fd6e4fa0980e446f5d999cce18da71e42bba` (36 layers, hidden 4096, 32
  query heads over 8 KV heads, intermediate 12288, vocabulary 151936; 15.27 GiB of weights).
- Owning modules: the recipe, `text_weights.zig` (q|k|v and gate|up fused into lane weights, as dots.mocr's loader),
  and the shared `qk_rows` kernel for each q and k head's RMS norm (`bf16(w * bf16(x * inv))`) and RoPE. No kernel is
  new or changed. Rotary frequencies are Transformers' own table for theta 1e6 at dim 128 (dots.mocr's fixture).
- Routing: `native/cuda.zig`, `qwen3HasHead`.
- Hardware: one NVIDIA L4 (sm_89), built with `zig build -Dgpu=sm_89 -Dnvcc=/usr/local/cuda/bin/nvcc` (CUDA 13.0,
  driver 580.178.04). sm_89 has no release image and is not model-qualified.

## Measured on one L4, 17 Nov 2026 (commit dca056d)

Speed with `bench.py` of the inference-optimisation skill (medians of 6), beside vLLM on the same card and checkpoint:

| | TensorFold | vLLM |
| --- | --- | --- |
| Load | 35.1 s | 136-184 s |
| One stream | 16.7 tok/s | 16.6 tok/s |
| Streams at once | 63.2 tok/s at 4 (the card's capacity at a 16,384-token window) | 696 tok/s at 64 |
| Prompt, 4,096 tokens | 1,782 tok/s | 3,123 tok/s |
| First token, short prompt | 125 ms | 74 ms |

Greedy output against vLLM on one 259-token prompt: the first 13 tokens are equal; the 14th differs where vLLM's two
best tokens are 0.125 apart in log probability (one bf16 step at that magnitude).

Task accuracy (FVEval-Verified NL-to-SVA, 4 sampled answers a task at temperature 0.8 and top-p 0.95, each judged by
an equivalence proof; 50 tasks so far): first answer proved on 39 tasks against vLLM's 41, any of four on 45 against
45, 161 of 200 answers proved against 162.

## Not done

- The exactness checks of [adding a Zig family](adding-a-zig-family.md) against a Transformers reference (first token,
  last-row logits within bf16 rounding, the next 48 tokens at 8k and 32k). The comparison above is against vLLM only.
- None of that guide's speed stages; the prompt pass is the dense engine's default.
- `n` above 1 and `logprobs` are refused by the server for this family, as for every dense recipe.
- `tensorfold info` reports 16,397,972,934 bytes for this folder: that is every file in it but one of 1,570 bytes
  (tokenizer, vocabulary and training logs included), not the tensors. The tensors are 16,381,470,720 bytes, summed
  from the safetensors headers, which is `text_weights.zig`'s `textBytes` exactly; the engine loads 15.61 GiB with
  its scratch.
- Host tests (`zig build test`) have not been run on this branch; only the sm_89 server build.
