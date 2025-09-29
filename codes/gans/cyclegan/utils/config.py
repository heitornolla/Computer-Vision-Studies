## hyper parameters
g_lr=0.0003                     # Learning Rate of Generator
d_lr=0.0001                     # Learning Rate of Discriminator
beta1=0.5                       # Beta 1 for Adam optimizer
beta2=0.999                     # Beta 2 for Adam optimizer
    
batch_size=2                    # batch size
init_epoch=0                    # initial epoch
n_epochs=500                    # maximum nomber of epochs
n_samples=-1                    # -1 for all available samples
test_size=0.5                   # fraction of all samples for validation
early_stop_epoch_thres=50       # threshold for stopping training if loss does not improve

image_size = (256, 256)         # image size 

# lambda for losses
lambda_discriminator=1.0        # lambda weight for discriminator loss
lambda_cycle_consistency=10.0   # lambda weight for cycle-consistency loss

# flags
use_pretrained_weights=False

# experiment id
experiment_id = 'cyclegan_improved'

## paths
domain_a_dir = 'a/*.png'
domain_b_dir = 'b/*.png'

# directories
checkpoints_dir = 'checkpoints'
samples_dir = 'samples'
logs_dir = 'logs'

# pretrained weights
generator_x_y_weights =    ''
generator_y_x_weights =    ''
discriminator_xp_weights = ''
discriminator_yp_weights = ''
discriminator_xg_weights = ''
discriminator_yg_weights = ''

## Number of epochs to sample and checkpoint
sample_every=1
checkpoint_every=10
