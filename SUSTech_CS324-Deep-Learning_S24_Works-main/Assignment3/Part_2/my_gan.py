import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
from tqdm import tqdm

DEFAULT_CONFIG = {
    'n_epochs': 200,
    'batch_size': 64,
    'lr': 0.0002,
    'latent_dim': 100,
    'save_interval': 500,
    'g_model_path': '',
    'save_path': 'images/train_latest',
}


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim

        # Construct generator. You should experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 784
        #   Output non-linearity

        self.network = nn.Sequential(
            nn.Linear(in_features=self.latent_dim, out_features=128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(in_features=128, out_features=256),
            nn.BatchNorm1d(num_features=256, momentum=0.8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(in_features=256, out_features=512),
            nn.BatchNorm1d(num_features=512, momentum=0.8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(in_features=512, out_features=1024),
            nn.BatchNorm1d(num_features=1024, momentum=0.8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(in_features=1024, out_features=784),

            nn.Tanh()
        )

    def forward(self, z):
        return self.network(z)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Construct distriminator. You should experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity

        self.network = nn.Sequential(
            nn.Linear(in_features=784, out_features=512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(in_features=256, out_features=1),

            nn.Sigmoid(),
        )

    def forward(self, img):
        return self.network(img)


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D, config):
    n_epochs = config['n_epochs']
    save_interval = config['save_interval']
    latent_dim = config['latent_dim']
    save_path = config['save_path']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.BCELoss()
    for epoch in tqdm(range(n_epochs)):
        for i, (imgs, _) in enumerate(dataloader):
            batch_size = imgs.size(0)
            real_labels = torch.ones(size=(batch_size, 1), device=device)
            fake_labels = torch.zeros(size=(batch_size, 1), device=device)

            real_imgs = imgs.view(batch_size, 1 * 28 * 28).to(device)

            z = torch.randn(size=(batch_size, latent_dim), device=device)

            # Train Generator
            generator.train()
            optimizer_G.zero_grad()
            # ---------------
            fake_imgs = generator(z)
            fake_outputs = discriminator(fake_imgs)
            g_loss = criterion(fake_outputs, real_labels)
            g_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            discriminator.train()
            optimizer_D.zero_grad()
            # -------------------

            # Real img
            real_outputs = discriminator(real_imgs)
            real_d_loss = criterion(real_outputs, real_labels)

            # Fake img
            fake_imgs_d = generator(z).detach()
            fake_outputs_d = discriminator(fake_imgs_d)
            fake_d_loss = criterion(fake_outputs_d, fake_labels)

            d_loss = real_d_loss + fake_d_loss
            d_loss.backward()
            optimizer_D.step()

            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % save_interval == 0:
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                fake_imgs = fake_imgs.view(batch_size, 1, 28, 28)
                save_image(fake_imgs[:25],
                           f'{save_path}{batches_done}.png',
                           nrow=5, normalize=True, value_range=(-1, 1))


def main(config):
    batch_size = config['batch_size']
    lr = config['lr']
    latent_dim = config['latent_dim']
    g_model_path = config['g_model_path']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create output image directory
    os.makedirs(config['save_path'], exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([transforms.ToTensor(),
                                                     transforms.Normalize(0.5, 0.5)])
                       ),
        batch_size=batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = Generator(latent_dim).to(device)
    if g_model_path != '':
        generator.load_state_dict(torch.load(g_model_path))
    discriminator = Discriminator().to(device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D, config)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    torch.save(generator.state_dict(), "./model/mnist_generator_latest.pt")


CONFIG = {
    'n_epochs': 200,
    'batch_size': 64,
    'lr': 0.0002,
    'latent_dim': 100,
    'save_interval': 500,
}

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--n_epochs', type=int, default=200,
    #                     help='number of epochs')
    # parser.add_argument('--batch_size', type=int, default=64,
    #                     help='batch size')
    # parser.add_argument('--lr', type=float, default=0.0002,
    #                     help='learning rate')
    # parser.add_argument('--latent_dim', type=int, default=100,
    #                     help='dimensionality of the latent space')
    # parser.add_argument('--save_interval', type=int, default=500,
    #                     help='save every SAVE_INTERVAL iterations')
    # args = parser.parse_args()
    # DEFAULT_CONFIG['n_epochs'] = args.n_epochs
    # DEFAULT_CONFIG['batch_size'] = args.batch_size
    # DEFAULT_CONFIG['lr'] = args.lr
    # DEFAULT_CONFIG['latent_dim'] = args.latent_dim
    # DEFAULT_CONFIG['save_interval'] = args.save_interval
    # seed = DEFAULT_CONFIG['seed']
    main(DEFAULT_CONFIG)
