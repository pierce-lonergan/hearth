# ADR-0005: The numpy reference is the numerical ground truth, cross-checked against transformers

- Status: Accepted
- Date: 2026-10-04
- Supersedes: none
- Superseded-by: none

## Context

The C engine needs a correctness oracle for five architecture families: Qwen3-MoE,
OLMoE, Mixtral, Qwen2-MoE and DeepSeek-V3/Kimi-K2 with MLA. Hugging Face
`transformers` defines what "correct" means for a released model. But it pulls
in torch, is slow on CPU, does not follow our canonical operation order, and
cannot read `.hearth` files. Comparing the engine directly with transformers
would mix up two separate questions: whether the conversion is right and
whether the kernels are right.

## Decision

* `hearth.reference` is a readable numpy forward pass over `.hearth`
  containers. It follows docs/NUMERICS.md and can emulate activation
  quantization. It is the ground truth for the engine:
  * logits within 1e-3 · max(1, max|logit|) on F32 containers (INV-NUM-1);
  * identical routing decisions (INV-NUM-2).
* The reference in turn must match `transformers` within 2e-3 relative for
  every supported architecture (INV-NUM-1). This is checked by
  `tests/py/test_convert_*.py`, which skip cleanly when torch or
  transformers are not installed.
* The converter computes RoPE frequencies and scaling with the source
  framework's own code, so the engine never has to implement YaRN, NTK or
  Llama-3 rope variants.
* numpy is the only hard Python dependency (INV-DEP).

## Consequences

* A disagreement is localised to one link: engine ↔ reference (kernels,
  scheduling) or reference ↔ transformers (conversion, architecture details).
* Default CI does not install torch. The transformers cross-check runs in an
  optional, manually triggered job.
* Adding an architecture means adding reference support, the converter
  mapping and a cross-check test, in that order.
