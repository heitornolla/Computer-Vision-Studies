## Autoencoders

Learn to compress and reconstruct data by mapping inputs to a latent (bottleneck) space and back.
- Latent space is unstructured: sampling from it typically does not produce meaningful outputs, because the model was not trained to generate from random latent vectors.
- No guarantee that similar points in latent space correspond to similar outputs.


## Variational Autoencoders

Introduce probabilistic inference; the encoder outputs a distribution (typically Gaussian) over latent variables.

- Latent space is continuous and normalized, encouraging smooth interpolation and structure.

Sampling is possible from a standard normal distribution, but:
- Reconstructions are often blurry.
- Generated samples can lack fine detail or realism.
