# Tensorized Embedding Layers

**Authors:** Oleksii Hrinchuk, Valentin Khrulkov, Leyla Mirvakhabova, Elena Orlova, Ivan Oseledets

## Abstract
The embedding layers transforming input words into real vectors are the key components of deep neural networks used in NLP. However, when the vocabulary is large, the corresponding weight matrices can be enormous. We introduce a novel way of parametrizing embedding layers based on the **Tensor Train (TT)** decomposition, which allows compressing the model significantly at negligible performance cost.

## Key Ideas
- Represent embedding weight matrix as a tensor in TT format.
- Compress embedding layers without performance degradation.
- Evaluated across MLPs, LSTMs, and Transformers.

## Excerpt
> "We evaluate our method on a wide range of NLP benchmarks and analyze the trade-off between performance and compression ratios for a wide range of architectures…"
