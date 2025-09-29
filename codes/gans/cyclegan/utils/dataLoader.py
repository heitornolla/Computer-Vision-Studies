import os
import glob
import cv2
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

class GAN_DataLoader:
    def __init__(self, imageX_dir: str, imageY_dir: str, image_size=(256, 256)):
        self.image_size = image_size
        self.imageX_dir = imageX_dir
        self.imageY_dir = imageY_dir

    def normalize_img(self, X: np.ndarray) -> np.ndarray:
        return (X.astype(np.float32) / 255.0) - 0.5

    def denormalize_img(self, X: np.ndarray) -> np.ndarray:
        return np.clip((X + 0.5) * 255.0, 0, 255).astype(np.uint8)

    def _get_fnames_list(self, n_samples=-1, test_size=0.1, shuffle=True):
        fnames_x = np.array(glob.glob(self.imageX_dir))
        fnames_y = np.array(glob.glob(self.imageY_dir))

        min_n_fnames = min(len(fnames_x), len(fnames_y))
        n_samples = min_n_fnames if n_samples == -1 else min(n_samples, min_n_fnames)

        if shuffle:
            idx = np.random.permutation(min_n_fnames)
            fnames_x, fnames_y = fnames_x[idx], fnames_y[idx]

        n_test = int(n_samples * test_size)
        n_train = n_samples - n_test

        return (
            fnames_x[:n_train],
            fnames_y[:n_train],
            fnames_x[n_train:n_samples],
            fnames_y[n_train:n_samples],
        )

    def _get_data_generator(self, fnames_x, fnames_y, batch_size=8, shuffle=True):
        assert len(fnames_x) == len(fnames_y)
        n_samples = len(fnames_x)
        while True:
            if shuffle:
                idx = np.random.permutation(n_samples)
                fnames_x, fnames_y = fnames_x[idx], fnames_y[idx]

            for offset in range(0, n_samples, batch_size):
                batch_x, batch_y = [], []
                bx = fnames_x[offset:offset + batch_size]
                by = fnames_y[offset:offset + batch_size]

                for fx, fy in zip(bx, by):
                    if os.path.exists(fx) and os.path.exists(fy):
                        img_x = cv2.cvtColor(cv2.imread(fx), cv2.COLOR_BGR2RGB)
                        img_x = cv2.resize(img_x, self.image_size)
                        img_x = self.normalize_img(img_x)
                        torch_img_x = np.moveaxis(img_x, -1, 0)

                        img_y = cv2.cvtColor(cv2.imread(fy), cv2.COLOR_BGR2RGB)
                        img_y = cv2.resize(img_y, self.image_size)
                        img_y = self.normalize_img(img_y)
                        torch_img_y = np.moveaxis(img_y, -1, 0)

                        batch_x.append(torch_img_x)
                        batch_y.append(torch_img_y)

                yield torch.tensor(batch_x, dtype=torch.float32), torch.tensor(batch_y, dtype=torch.float32)

    def get_data_generator(self, n_samples=-1, test_size=0.1, batch_size=8, shuffle=True):
        fnames_x_train, fnames_y_train, fnames_x_test, fnames_y_test = self._get_fnames_list(
            n_samples=n_samples, test_size=test_size, shuffle=True
        )
        return (
            self._get_data_generator(fnames_x_train, fnames_y_train, batch_size=batch_size, shuffle=shuffle),
            self._get_data_generator(fnames_x_test, fnames_y_test, batch_size=batch_size, shuffle=shuffle),
        )

    def get_num_samples(self, n_samples=-1, test_size=0.1):
        fnames_x_train, _, fnames_x_test, _ = self._get_fnames_list(
            n_samples=n_samples, test_size=test_size, shuffle=False
        )
        return len(fnames_x_train), len(fnames_x_test)

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

if __name__ == '__main__':
    dloader = GAN_DataLoader(
        imageX_dir='',
        imageY_dir=''
    )
    dloader_train, dloader_test = dloader.get_data_generator()
    dloader_test_it = iter(dloader_test)
    images_x, images_y = next(dloader_test_it)
    print(images_y.min().item(), images_y.max().item())

    fig = plt.figure(figsize=(12, 12))
    plt.subplot(2, 1, 1)
    images_x = images_x + 0.5
    imshow(torchvision.utils.make_grid(images_x, nrow=4))
    plt.title('Domain A')

    plt.subplot(2, 1, 2)
    images_y = images_y + 0.5
    imshow(torchvision.utils.make_grid(images_y, nrow=4))
    plt.title('Domain B')
    plt.show()

