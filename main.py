import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import utils
from GAN_model import GeneratorDcGan, DiscriminatorDcGan
from GAN_utils import get_mixed_image_gradient, get_gradient_penalty, get_crit_loss, get_gen_loss
import GAN_settings

if __name__ == '__main__':

    DEVICE = utils.get_device()

    print('Running DCGAN')

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    # trainset = FashionMNIST('.', download=True, train=True, transform=transform)
    trainset = MNIST('.', download=True, train=True, transform=transform)

    data_loader = DataLoader(trainset, batch_size=GAN_settings.BATCH_SIZE)

    print(f'Num images = {len(data_loader)}')

    generator_model = GeneratorDcGan().to(DEVICE)
    discriminator_model = DiscriminatorDcGan().to(DEVICE)

    utils.get_torch_model_output_size_at_each_layer(generator_model, (1, generator_model.in_noise_dim, 1, 1))

    print(generator_model)

    gen_optimizer = torch.optim.Adam(generator_model.parameters(), lr=GAN_settings.GEN_LEARNING_RATE)
    disc_optimizer = torch.optim.Adam(discriminator_model.parameters(), lr=GAN_settings.DISC_LEARNING_RATE)

    generator_model.load_state_dict(torch.load(GAN_settings.MODEL_SAVE_PATH + f'gen_model_{8}_{100}'))
    discriminator_model.load_state_dict(torch.load(GAN_settings.MODEL_SAVE_PATH + f'disc_model_{8}_{100}'))
    for epoch in range(GAN_settings.EPOCHS):

        print(f'EPOCH = {epoch} -------------------- ')

        generator_model.train()
        discriminator_model.train()

        step = 0
        for real_images, _ in data_loader:
            # real_image shape = [batch_size, 1, 28, 28]
            batch_size = real_images.shape[0]
            real_images = real_images.to(DEVICE)

            ##########################
            # CRITIC TRAINING        #
            ##########################

            for _ in range(GAN_settings.NUM_CRITIC_TRAIN_LOOP):
                disc_optimizer.zero_grad()
                generator_model.eval()

                # Create the fake image
                in_noise = generator_model.get_noise(batch_size, generator_model.in_noise_dim).to(DEVICE)
                fake_images = generator_model(in_noise)

                fake_image_disc_pred = discriminator_model(fake_images.detach())
                real_image_disc_pred = discriminator_model(real_images)

                # epsilon shape = (num images, 1, 1, 1)
                epsilon = torch.rand(batch_size, 1, 1, 1, device=DEVICE, requires_grad=True)
                gradient = get_mixed_image_gradient(discriminator_model, real_images, fake_images.detach(), epsilon)
                gradient_penalty = get_gradient_penalty(gradient)
                crit_loss = get_crit_loss(fake_image_disc_pred, real_image_disc_pred, gradient_penalty, GAN_settings.C_LAMBDA)

                crit_loss.backward(retain_graph=True)
                disc_optimizer.step()

                # print(f'total_disc_loss = {crit_loss}')

            ######################
            # GENERATOR TRAINING #
            ######################

            for _ in range(GAN_settings.NUM_GENERATOR_TRAIN_LOOP):
                gen_optimizer.zero_grad()
                generator_model.train()

                # Create the fake image
                in_noise_2 = generator_model.get_noise(batch_size, generator_model.in_noise_dim).to(DEVICE)
                fake_images_2 = generator_model(in_noise_2)

                # we should not detach fake_images_2 as the gradients have to flow from disc to gen model
                fake_image_disc_pred_2 = discriminator_model(fake_images_2)
                gen_loss_fake_image = get_gen_loss(fake_image_disc_pred_2)

                gen_loss_fake_image.backward()

                gen_optimizer.step()

                # print(f'gen_loss_fake_image = {gen_loss_fake_image}')

            if step % 50 == 0:
                print(f'total_disc_loss = {crit_loss}  gen_loss_fake_image = {gen_loss_fake_image}')
                utils.save_tensor_images(fake_images, file_name=f'./output/fake_{epoch}_{step}.png')
                torch.save(discriminator_model.state_dict(),
                           GAN_settings.MODEL_SAVE_PATH + f'disc_model_{epoch}_{step}')
                torch.save(generator_model.state_dict(), GAN_settings.MODEL_SAVE_PATH + f'gen_model_{epoch}_{step}')
                # utils.save_tensor_images(real_images, file_name='real.png')

            step += 1
