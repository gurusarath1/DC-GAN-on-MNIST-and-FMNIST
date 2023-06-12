import numpy as np
import torch
import torch.nn as nn
import GAN_settings


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


class GeneratorDcGan(nn.Module):

    def __init__(self, in_noise_dim=62, output_image_channels=1, hidden_dim=100):
        super(GeneratorDcGan, self).__init__()

        self.in_noise_dim = in_noise_dim

        self.gen_model = nn.Sequential(
            self.dc_gan_gen_block(self.in_noise_dim, hidden_dim * 4),
            self.dc_gan_gen_block(hidden_dim * 4, hidden_dim * 2, kernal_size=4, stride=1),
            self.dc_gan_gen_block(hidden_dim * 2, hidden_dim),

            # These set of layers does not increase or decrease the filter size
            nn.ConvTranspose2d(100, 100, (1, 1), (1, 1)),
            nn.BatchNorm2d(100),
            nn.ReLU(),

            # Last block
            self.dc_gan_last_gen_block(hidden_dim, output_image_channels, kernal_size=4)
        )

        self.gen_model.apply(weights_init)

    def dc_gan_gen_block(self, in_channels, out_channels, kernal_size=3, stride=2):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernal_size, stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def dc_gan_last_gen_block(self, in_channels, out_channels, kernal_size=3, stride=2):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernal_size, stride),
            nn.Tanh()
        )

    def process_input_noise_for_model_input(self, noise):
        assert noise.shape[1] == self.in_noise_dim

        batch_size = noise.shape[0]
        return noise.view(batch_size, self.in_noise_dim, 1, 1)  # (batch_size, channels, height, width)

    def forward(self, noise):
        input_noise_batch = self.process_input_noise_for_model_input(noise)

        return self.gen_model(input_noise_batch)

    @staticmethod
    def get_noise(num_samples, noise_dim, device=GAN_settings.DEVICE):
        return torch.randn(num_samples, noise_dim).to(device)


class DiscriminatorDcGan(nn.Module):

    def __init__(self, input_image_channels=1, hidden_dim=16):
        super(DiscriminatorDcGan, self).__init__()

        self.input_image_channels = input_image_channels

        disc_out_channels = 1

        self.disc_model = nn.Sequential(
            self.dc_gan_disc_block(input_image_channels, hidden_dim),
            self.dc_gan_disc_block(hidden_dim, hidden_dim * 2),
            self.dc_gan_disc_block(hidden_dim * 2, disc_out_channels, last_block=True)
        )

        self.disc_model.apply(weights_init)

    def dc_gan_disc_block(self, in_channels, out_channels, kernel_size=4, stride=2, last_block=False):
        if last_block:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            )

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, image):
        out = self.disc_model(image)
        batch_size = out.shape[0]
        return out.view(batch_size, -1)  # Flattened output


