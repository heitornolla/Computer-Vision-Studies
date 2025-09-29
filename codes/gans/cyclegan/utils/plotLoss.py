import csv
import matplotlib.pyplot as plt

CSV_LOG_FILE = './logs/cyclegan_improved/logs.csv'
MAX_EPOCHS = 100

def plot_losses():
    epoch = []
    d_X_loss, d_Y_loss, recon_X_loss, recon_Y_loss, total_loss = [], [], [], [], []
    valid_d_X_loss, valid_d_Y_loss, valid_recon_X_loss, valid_recon_Y_loss, valid_total_loss = [], [], [], [], []

    with open(CSV_LOG_FILE, newline='') as file:
        reader = csv.DictReader(file)
        for line in reader:
            e = int(line['epoch'])
            epoch.append(e)
            d_X_loss.append(float(line['d_X_loss']))
            d_Y_loss.append(float(line['d_Y_loss']))
            recon_X_loss.append(float(line['recon_X_loss']))
            recon_Y_loss.append(float(line['recon_Y_loss']))
            total_loss.append(float(line['total_loss']))
            valid_d_X_loss.append(float(line['valid_d_X_loss']))
            valid_d_Y_loss.append(float(line['valid_d_Y_loss']))
            valid_recon_X_loss.append(float(line['valid_recon_X_loss']))
            valid_recon_Y_loss.append(float(line['valid_recon_Y_loss']))
            valid_total_loss.append(float(line['valid_total_loss']))

            if e == MAX_EPOCHS:
                break

    fig, ax = plt.subplots(2, figsize=(12, 12))
    ax[0].set_title('Training Losses', fontweight='bold')
    ax[0].grid(linestyle='-', linewidth=0.2, color='gray')
    ax[0].plot(epoch, d_X_loss)
    ax[0].plot(epoch, d_Y_loss)
    ax[0].plot(epoch, recon_X_loss)
    ax[0].plot(epoch, recon_Y_loss)
    ax[0].plot(epoch, total_loss)
    ax[0].legend([
        'x_discriminator_loss', 'y_discriminator_loss', 'x_reconstruction_loss', 'y_reconstruction_loss', 'total_loss'
    ], loc='upper right', fancybox=True, framealpha=1., shadow=True, borderpad=1)
    ax[0].set_ylabel('Losses', fontweight='bold')

    ax[1].set_title('Validation Losses', fontweight='bold')
    ax[1].grid(linestyle='-', linewidth=0.2, color='gray')
    ax[1].plot(epoch, valid_d_X_loss)
    ax[1].plot(epoch, valid_d_Y_loss)
    ax[1].plot(epoch, valid_recon_X_loss)
    ax[1].plot(epoch, valid_recon_Y_loss)
    ax[1].plot(epoch, valid_total_loss)
    ax[1].legend([
        'x_discriminator_loss', 'y_discriminator_loss', 'x_reconstruction_loss', 'y_reconstruction_loss', 'total_loss'
    ], loc='upper right', fancybox=True, framealpha=1., shadow=True, borderpad=1)
    ax[1].set_ylabel('Losses', fontweight='bold')

    plt.savefig('media/losses.png')
    plt.show()

if __name__ == '__main__':
    plot_losses()

