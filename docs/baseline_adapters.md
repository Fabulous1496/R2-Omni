# Omni model adapters

KIVI and OTT do not provide native Qwen2.5-Omni or MiniCPM-o integration. The adapters in
this repository leave their quantization algorithms and packed CUDA kernels
unchanged and implement only the model boundary:

- Qwen Q/K/V projections and Qwen2.5 mRoPE or Qwen3 RoPE/QK-Norm;
- GQA head mapping and cache layout conversion;
- packed-cache creation after FA2 prefill;
- packed and Recent-FP attention during decode.

Qwen2.5-Omni-7B uses FP16 host states and pre-scales queries before the QK
readout. This is algebraically equivalent to scaling the completed dot product
and avoids FP16 intermediate overflow in deep layers. The upstream kernels are
unchanged.

The default launchers use native packed decode. Dense FA2 and SDPA readouts are
available in the adapter as correctness references, but materialize K/V and
must not be used to report packed-kernel latency or memory.

MiniCPM-o-4.5 keeps its official audiovisual preprocessing and chat lifecycle.
Its LLM and vision prefills use FA2, while the upstream audio encoder remains
SDPA. The integration intercepts only the call to the internal Qwen3 LLM.
OScaR uses its released Qwen3 attention and CUDA cache; KIVI-OM and OTT-OM use
their upstream quantizers and packed GEMV. OScaR must be compiled against the
same PyTorch ABI as MiniCPM rather than loaded from an older binary build. A
thin cache-type bridge lets Transformers generation recognize OScaR's cache;
the cache transitions and CUDA readout remain OScaR implementations. The
adapter also applies OScaR's released offline V/O Hadamard weight transform
before INT2 cache construction.
