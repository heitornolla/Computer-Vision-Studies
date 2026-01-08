import random
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import glob

from PIL import Image
from utils.utils import PadOrCropPIL

import utils.config as config

import numpy as np
from torch.utils.data import Dataset, DataLoader


class GAN_DataLoader(Dataset):
    def __init__(
        self, ref_imgs, lat_imgs, image_size=config.IMG_SIZE, apply_transforms=True
    ):
        self.ref_imgs = ref_imgs
        self.lat_imgs = lat_imgs
        self.apply_transforms = apply_transforms

        base_transform = [
            PadOrCropPIL(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]

        if self.apply_transforms:
            # Stage 2 from paper: Random shifts and rotations
            # fill=255 assumes white background for fingerprints
            augmentations = [
                transforms.RandomAffine(degrees=15, translate=(0.2, 0.2), fill=255),
                transforms.RandomHorizontalFlip(),
            ]
            self.transform = transforms.Compose(augmentations + base_transform)
        else:
            self.transform = transforms.Compose(base_transform)

    def __len__(self):
        return max(len(self.ref_imgs), len(self.lat_imgs))

    def __getitem__(self, index):
        img_a_path = self.ref_imgs[index % len(self.ref_imgs)]
        img_b_path = self.lat_imgs[index % len(self.lat_imgs)]

        img_a = Image.open(img_a_path).convert("RGB")
        img_b = Image.open(img_b_path).convert("RGB")

        return self.transform(img_a), self.transform(img_b)


def get_fingerprint_loaders():
    all_refs = sorted(glob.glob(config.REF_IMGS_PATH))
    all_lats = sorted(glob.glob(config.LAT_IMGS_PATH))

    random.seed(42)
    random.shuffle(all_refs)
    random.shuffle(all_lats)

    split_refs = int(len(all_refs) * (1 - config.TEST_SIZE))
    split_lats = int(len(all_lats) * (1 - config.TEST_SIZE))

    train_fnames_ref, test_fnames_ref = all_refs[:split_refs], all_refs[split_refs:]
    train_fnames_lat, test_fnames_lat = all_lats[:split_lats], all_lats[split_lats:]

    train_dataset = GAN_DataLoader(
        train_fnames_ref,
        train_fnames_lat,
        image_size=config.IMG_SIZE,
        apply_transforms=True,
    )

    test_dataset = GAN_DataLoader(
        test_fnames_ref,
        test_fnames_lat,
        image_size=config.IMG_SIZE,
        apply_transforms=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, test_loader


# helper imshow function
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


if __name__ == "__main__":
    dloader_train, dloader_test = get_fingerprint_loaders()
    dloader_train_it = iter(dloader_train)
    dloader_test_it = iter(dloader_test)

    # the "_" is a placeholder for no labels
    images_x, images_y = next(dloader_test_it)
    print(images_y.min(), images_y.max())

    # show images
    fig = plt.figure(figsize=(12, 12))
    plt.subplot(2, 1, 1)
    images_x = images_x + 0.5
    imshow(torchvision.utils.make_grid(images_x, nrow=4))
    plt.title("Domain A")
    plt.subplot(2, 1, 2)
    images_y = images_y + 0.5
    imshow(torchvision.utils.make_grid(images_y, nrow=4))
    plt.title("Domain B")
    plt.show()
