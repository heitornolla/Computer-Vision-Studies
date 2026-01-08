import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256
REF_IMGS_PATH = "/home/heitor/LIDS/datasets/sd302/reference/*/*.png"
LAT_IMGS_PATH = "/home/heitor/LIDS/datasets/sd302/latents/*.png"
BATCH_SIZE = 8
GEN_LEARNING_RATE = 0.0003
DISC_LEARNING_RATE = 0.0001
LAMBDA_IDENTITY = 5
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 10
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_REF = "gen_r.pth.tar"
CHECKPOINT_GEN_LAT = "gen_l.pth.tar"
CHECKPOINT_DISC_REF = "disc_r.pth.tar"
CHECKPOINT_DISC_LAT = "disc_l.pth.tar"
TEST_SIZE = 0.2
