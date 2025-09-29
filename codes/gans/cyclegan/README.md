# [CycleGAN](https://junyanz.github.io/CycleGAN/), and its incremental improvement

This is a simple PyTorch implementation of [CycleGAN](https://junyanz.github.io/CycleGAN/) and a study of its incremental improvements.

CycleGAN provided a huge insight into the idea of cycle-consistency for domain adaptation. It initially proposed using a cycle-consistency loss combined with the adversarial loss to ensure that domain-translated image looks realistic while maintaining forward and backward consistency of the image. This was a great help towards solving the mode collapse problem and it also maintains the shape of objects within scene. The discriminator was proposed to output a single fake/real prediction, which since then have been proposed to be replaced with a patch discriminator. Patch discriminator looks at a small patch of the generator, say 16x16 patch, and classifies it as real/fake. This forces the generator to get the domain translation right down to smaller scale.

This code has been adapted to work with Pytorch 2.7+

## Key Details
 - A combination of Patch Discriminator and Global Discriminator is used.  
 - Different Learning Rate is used for Generator and Discriminator. Discriminator uses a lower learning rate.    
 - Discriminator's learning is disabled during the training of Generator.   
 - Leaky Relu with a negative slope of 0.2 is used in Generator, but ReLU is used in Discriminator.  
 - Instance Normalization is used.  
 - Output of the Generator is clamped to output a value between [-0.5, 0.5].  
