# Terms used in this experiment

| Term | Definition |
|---|---|
| AMP-FP16 baseline | Same checkpoint and test path without HQQ, evaluated under CUDA autocast FP16. |
| Weight-only | HQQ packed Linear weights at W3 or W4 and weight group size 8 or 128; classification head excluded. |
| Linear activation | HQQ fork `fake_quant_activation` on each quantized Linear input, A8 or A4, using its final dimension per token. |
| QKV activation | Additional fake quantization of Q, K and V after head splitting, with scale per token/head along head dimension; K is quantized before transpose. |
| Relative accuracy loss | `1 - quant_accuracy / FP_accuracy`; undefined when FP accuracy is zero. |
| Accuracy drop | `FP_accuracy - quant_accuracy` in fractional accuracy units. |
| Swap rate | Fraction whose top-1 prediction differs from FP. |
| Damage / rescue | FP correct to quantized wrong / FP wrong to quantized correct, respectively. |
| Net damage R | `(damage - rescue) / swaps`; undefined when no predictions swap. |
| Centered linear CKA | Feature-space similarity of matched, centered FP and quantized representations at the same layer. |
| Class-equal separation | Cosine `D_inter - D_intra` after feature L2 normalization, averaging classes and class pairs equally. |
| Regularized H-score | Trace of between-class scatter against ridge-regularized within-class scatter. |
| Normalized margin Q10 | 10th percentile of each sample's top-1 minus top-2 logit gap divided by its logit standard deviation. |
| FP-fitted PCA | One 2D basis fitted to FP final CLS per dataset, reused to project all quantized final CLS features. |

Activation quantization here is a fake quantization precision experiment. Its floating-point operations and packed HQQ weights do not establish integer-kernel throughput or latency improvement.
