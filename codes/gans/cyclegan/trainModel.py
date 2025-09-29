import os
import torch
import torch.optim as optim
from torchsummary import summary
import matplotlib.pyplot as plt
from collections import deque
import math
import time
import csv
import numpy as np

from dataLoader import GAN_DataLoader
from model import CycleGAN
from lossFunc import *
from utils import *
from config import *

# configure full paths
checkpoints_dir = os.path.join(checkpoints_dir, experiment_id)
samples_dir = os.path.join(samples_dir, experiment_id)
logs_dir = os.path.join(logs_dir, experiment_id)

# make directories safely
os.makedirs(checkpoints_dir, exist_ok=True)
os.makedirs(samples_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# create models
G_XtoY, G_YtoX, Dp_X, Dp_Y, Dg_X, Dg_Y = CycleGAN(n_res_blocks=2)

# optimizer params
g_params = list(G_XtoY.parameters()) + list(G_YtoX.parameters())

# optimizers
g_optimizer = optim.Adam(g_params, lr=g_lr, betas=(beta1, beta2))
dp_x_optimizer = optim.Adam(Dp_X.parameters(), lr=d_lr, betas=(beta1, beta2))
dp_y_optimizer = optim.Adam(Dp_Y.parameters(), lr=d_lr, betas=(beta1, beta2))
dg_x_optimizer = optim.Adam(Dg_X.parameters(), lr=d_lr, betas=(beta1, beta2))
dg_y_optimizer = optim.Adam(Dg_Y.parameters(), lr=d_lr, betas=(beta1, beta2))

def count_model_parameters(model):
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_params, n_trainable_params, n_params - n_trainable_params

def trainModel(dloader_train_it, dloader_test_it, n_train_batch_per_epoch, n_test_batch_per_epoch, batch_size=8, n_epochs=1000):
    n_consecutive_epochs = 0
    mean_epochs = 10
    losses = []
    min_loss = 1.0e6

    fixed_X, fixed_Y = next(dloader_test_it)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_file = os.path.join(logs_dir, 'log.csv')
    logfile_exists = os.path.exists(log_file)

    with open(log_file, 'a+', newline='') as csvfile:
        fieldnames = [
            'epoch', 'd_X_loss', 'd_Y_loss', 'recon_X_loss', 'recon_Y_loss', 'total_loss',
            'valid_d_X_loss', 'valid_d_Y_loss', 'valid_recon_X_loss', 'valid_recon_Y_loss', 'valid_total_loss'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not logfile_exists:
            writer.writeheader()

        for epoch in range(init_epoch, n_epochs):
            train_losses = deque(maxlen=mean_epochs)
            start_time = time.time()

            for iteration in range(n_train_batch_per_epoch):
                images_X, images_Y = next(dloader_train_it)
                images_X = images_X.to(device, non_blocking=True)
                images_Y = images_Y.to(device, non_blocking=True)

                # === Train Discriminators ===
                Dp_X.train(); Dp_Y.train(); Dg_X.train(); Dg_Y.train()

                dp_x_optimizer.zero_grad(set_to_none=True)
                dg_x_optimizer.zero_grad(set_to_none=True)
                out_xp, out_xg = Dp_X(images_X), Dg_X(images_X)
                dp_xreal_loss = real_discriminator_loss(out_xp, lambda_weight=lambda_discriminator)
                dg_xreal_loss = real_discriminator_loss(out_xg, lambda_weight=lambda_discriminator)
                D_X_real_loss = dp_xreal_loss + dg_xreal_loss

                fake_X = G_YtoX(images_Y)
                out_xp, out_xg = Dp_X(fake_X.detach()), Dg_X(fake_X.detach())
                dp_xfake_loss = fake_discriminator_loss(out_xp, lambda_weight=lambda_discriminator)
                dg_xfake_loss = fake_discriminator_loss(out_xg, lambda_weight=lambda_discriminator)
                D_X_fake_loss = dp_xfake_loss + dg_xfake_loss

                d_x_loss = D_X_real_loss + D_X_fake_loss
                d_x_loss.backward()
                dp_x_optimizer.step(); dg_x_optimizer.step()

                dp_y_optimizer.zero_grad(set_to_none=True)
                dg_y_optimizer.zero_grad(set_to_none=True)
                out_yp, out_yg = Dp_Y(images_Y), Dg_Y(images_Y)
                dp_yreal_loss = real_discriminator_loss(out_yp, lambda_weight=lambda_discriminator)
                dg_yreal_loss = real_discriminator_loss(out_yg, lambda_weight=lambda_discriminator)
                D_Y_real_loss = dp_yreal_loss + dg_yreal_loss

                fake_Y = G_XtoY(images_X)
                out_yp, out_yg = Dp_Y(fake_Y.detach()), Dg_Y(fake_Y.detach())
                dp_yfake_loss = fake_discriminator_loss(out_yp, lambda_weight=lambda_discriminator)
                dg_yfake_loss = fake_discriminator_loss(out_yg, lambda_weight=lambda_discriminator)
                D_Y_fake_loss = dp_yfake_loss + dg_yfake_loss

                dp_loss = 0.25 * (dp_xreal_loss + dp_xfake_loss + dp_yreal_loss + dp_yfake_loss)
                dg_loss = 0.25 * (dg_xreal_loss + dg_xfake_loss + dg_yreal_loss + dg_yfake_loss)

                d_y_loss = 0.5 * (D_Y_real_loss + D_Y_fake_loss)
                d_y_loss.backward()
                dp_y_optimizer.step(); dg_y_optimizer.step()

                # === Train Generators ===
                G_YtoX.train(); G_XtoY.train()
                Dp_X.eval(); Dp_Y.eval(); Dg_X.eval(); Dg_Y.eval()
                g_optimizer.zero_grad(set_to_none=True)

                fake_X = G_YtoX(images_Y)
                g_YtoX_loss = real_discriminator_loss(Dp_X(fake_X), lambda_weight=lambda_discriminator) + \
                              real_discriminator_loss(Dg_X(fake_X), lambda_weight=lambda_discriminator)
                reconstructed_Y = G_XtoY(fake_X)
                reconstructed_y_loss = cycle_consistency_loss(images_Y, reconstructed_Y, lambda_weight=lambda_cycle_consistency)

                fake_Y = G_XtoY(images_X)
                g_XtoY_loss = real_discriminator_loss(Dp_Y(fake_Y), lambda_weight=lambda_discriminator) + \
                              real_discriminator_loss(Dg_Y(fake_Y), lambda_weight=lambda_discriminator)
                reconstructed_X = G_YtoX(fake_Y)
                reconstructed_x_loss = cycle_consistency_loss(images_X, reconstructed_X, lambda_weight=lambda_cycle_consistency)

                g_total_loss = g_XtoY_loss + g_YtoX_loss + reconstructed_x_loss + reconstructed_y_loss
                g_total_loss.backward()
                g_optimizer.step()

                train_losses.append([
                    d_x_loss.item(), d_y_loss.item(), reconstructed_x_loss.item(), reconstructed_y_loss.item(), g_total_loss.item()
                ])

                print(f"\rEpoch [{epoch:5d}/{n_epochs:5d}] | Iter [{iteration:5d}/{n_train_batch_per_epoch:5d}] | d_X_loss: {d_x_loss.item():6.4f} | d_Y_loss: {d_y_loss.item():6.4f} | dp_loss: {dp_loss.item():6.4f} | dg_loss: {dg_loss.item():6.4f} | recon_X_loss: {reconstructed_x_loss.item():6.4f} | recon_Y_loss: {reconstructed_y_loss.item():6.4f} | total_loss: {g_total_loss.item():6.4f}", end="")

            time_taken = time.time() - start_time

            # === Validation ===
            validation_losses = deque(maxlen=mean_epochs)
            with torch.inference_mode():
                for _ in range(n_test_batch_per_epoch):
                    images_X, images_Y = next(dloader_test_it)
                    images_X = images_X.to(device, non_blocking=True)
                    images_Y = images_Y.to(device, non_blocking=True)

                    G_YtoX.eval(); G_XtoY.eval(); Dp_X.eval(); Dp_Y.eval(); Dg_X.eval(); Dg_Y.eval()

                    fake_Y = G_XtoY(images_X)
                    recon_Y_X = G_YtoX(fake_Y)
                    reconstructed_X_loss = cycle_consistency_loss(images_X, recon_Y_X, lambda_weight=lambda_cycle_consistency)

                    fake_X = G_YtoX(images_Y)
                    recon_X_Y = G_XtoY(fake_X)
                    reconstructed_Y_loss = cycle_consistency_loss(images_Y, recon_X_Y, lambda_weight=lambda_cycle_consistency)

                    disc_X_loss = real_discriminator_loss(Dp_X(images_X), lambda_weight=lambda_discriminator) + \
                                   real_discriminator_loss(Dg_X(images_X), lambda_weight=lambda_discriminator) + \
                                   fake_discriminator_loss(Dp_X(fake_X), lambda_weight=lambda_discriminator) + \
                                   fake_discriminator_loss(Dg_X(fake_X), lambda_weight=lambda_discriminator)

                    disc_Y_loss = real_discriminator_loss(Dp_Y(images_Y), lambda_weight=lambda_discriminator) + \
                                   real_discriminator_loss(Dg_Y(images_Y), lambda_weight=lambda_discriminator) + \
                                   fake_discriminator_loss(Dp_Y(fake_Y), lambda_weight=lambda_discriminator) + \
                                   fake_discriminator_loss(Dg_Y(fake_Y), lambda_weight=lambda_discriminator)

                    total_valid_loss = reconstructed_X_loss + reconstructed_Y_loss + disc_X_loss + disc_Y_loss
                    validation_losses.append([
                        disc_X_loss.item(), disc_Y_loss.item(), reconstructed_X_loss.item(), reconstructed_Y_loss.item(), total_valid_loss.item()
                    ])

            # averages
            train_losses_np = np.array(train_losses)
            train_d_X_loss, train_d_Y_loss, train_recon_X_loss, train_recon_Y_loss, train_total_G_loss = train_losses_np.mean(axis=0)
            total_recon_loss = train_recon_X_loss + train_recon_Y_loss

            valid_losses_np = np.array(validation_losses)
            valid_d_X_loss, valid_d_Y_loss, valid_recon_X_loss, valid_recon_Y_loss, valid_total_G_loss = valid_losses_np.mean(axis=0)

            print(f"\nEpoch [{epoch:5d}/{n_epochs:5d}] Training Losses   | d_X_loss: {train_d_X_loss:6.4f} | d_Y_loss: {train_d_Y_loss:6.4f} | recon_X_loss: {train_recon_X_loss:6.4f} | recon_Y_loss: {train_recon_Y_loss:6.4f} | total_loss: {train_total_G_loss:6.4f} | Time Taken: {time_taken:.2f} sec")
            print(f"Epoch [{epoch:5d}/{n_epochs:5d}] Validation Losses | d_X_loss: {valid_d_X_loss:6.4f} | d_Y_loss: {valid_d_Y_loss:6.4f} | recon_X_loss: {valid_recon_X_loss:6.4f} | recon_Y_loss: {valid_recon_Y_loss:6.4f} | total_loss: {valid_total_G_loss:6.4f}")

            if epoch % sample_every == 0:
                G_YtoX.eval(); G_XtoY.eval()
                save_samples(samples_dir, epoch, fixed_Y, fixed_X, G_YtoX, G_XtoY, batch_size=batch_size)
                G_YtoX.train(); G_XtoY.train()

            if epoch % checkpoint_every == 0:
                checkpoint(checkpoints_dir, epoch, G_XtoY, G_YtoX, Dp_X, Dp_Y, Dg_X, Dg_Y)

            if total_recon_loss <= min_loss:
                print(f'Total Reconstruction Loss decreased from {min_loss:.5f} to {total_recon_loss:.5f}. Saving Model.')
                checkpoint(checkpoints_dir, epoch, G_XtoY, G_YtoX, Dp_X, Dp_Y, Dg_X, Dg_Y, best=True)
                min_loss = total_recon_loss
                n_consecutive_epochs = 0
            else:
                print(f'Total Reconstruction Loss did not improve from {min_loss:.5f}')
                n_consecutive_epochs += 1

            writer.writerow({
                'epoch': epoch,
                'd_X_loss': train_d_X_loss,
                'd_Y_loss': train_d_Y_loss,
                'recon_X_loss': train_recon_X_loss,
                'recon_Y_loss': train_recon_Y_loss,
                'total_loss': train_total_G_loss,
                'valid_d_X_loss': valid_d_X_loss,
                'valid_d_Y_loss': valid_d_Y_loss,
                'valid_recon_X_loss': valid_recon_X_loss,
                'valid_recon_Y_loss': valid_recon_Y_loss,
                'valid_total_loss': valid_total_G_loss
            })

            if n_consecutive_epochs >= early_stop_epoch_thres:
                print(f'Total Loss did not improve for {n_consecutive_epochs} consecutive epochs. Early Stopping!')
                n_epochs = epoch
                break

    checkpoint(checkpoints_dir, n_epochs, G_XtoY, G_YtoX, Dp_X, Dp_Y, Dg_X, Dg_Y)
    print(f'Training completed in {n_epochs} epochs.')

if __name__ == '__main__':
    print('='*58)
    print('=====================  Model Summary  ====================')
    print('='*58)
    for name, model in [('G_XtoY', G_XtoY), ('G_YtoX', G_YtoX), ('Dp_X', Dp_X), ('Dp_Y', Dp_Y)]:
        n_params, n_trainable_params, n_non_trainable_params = count_model_parameters(model)
        print(f'{name}:')
        print(f'\t- Num of Parameters                : {n_params:,}')
        print(f'\t- Num of Trainable Parameters      : {n_trainable_params:,}')
        print(f'\t- Num of Non-Trainable Parameters  : {n_non_trainable_params:,}')
    print('='*58)
    summary(G_XtoY, (3, image_size[1], image_size[0]))
    summary(Dp_X, (3, image_size[1], image_size[0]))
    summary(Dg_X, (3, image_size[1], image_size[0]))

    if use_pretrained_weights:
        G_XtoY.load_state_dict(torch.load(generator_x_y_weights, map_location='cpu'))
        G_YtoX.load_state_dict(torch.load(generator_y_x_weights, map_location='cpu'))
        Dp_X.load_state_dict(torch.load(discriminator_xp_weights, map_location='cpu'))
        Dp_Y.load_state_dict(torch.load(discriminator_yp_weights, map_location='cpu'))
        Dg_X.load_state_dict(torch.load(discriminator_xg_weights, map_location='cpu'))
        Dg_Y.load_state_dict(torch.load(discriminator_yg_weights, map_location='cpu'))
        print('Loaded pretrained weights')

    dloader = GAN_DataLoader(imageX_dir=domain_a_dir, imageY_dir=domain_b_dir, image_size=image_size)
    dloader_train, dloader_test = dloader.get_data_generator(n_samples=n_samples, test_size=test_size, batch_size=batch_size)
    dloader_train_it, dloader_test_it = iter(dloader_train), iter(dloader_test)

    n_train_samples, n_test_samples = dloader.get_num_samples(n_samples=n_samples, test_size=test_size)
    n_train_batch_per_epoch = math.ceil(n_train_samples / batch_size)
    n_test_batch_per_epoch = math.ceil(n_test_samples / batch_size)
    print(f'Training on {n_train_samples} samples and testing on {n_test_samples} samples for a maximum of {n_epochs} epochs.')

    trainModel(dloader_train_it, dloader_test_it, n_train_batch_per_epoch, n_test_batch_per_epoch, batch_size=batch_size, n_epochs=n_epochs)
