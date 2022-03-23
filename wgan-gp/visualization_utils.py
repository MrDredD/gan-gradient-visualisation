import torch
import utils
import config
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from models import Generator, Critic, initialize_weights
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image


def load_models(epoch):
    model_gen = Generator(config.Z_DIM, config.IMG_CHANNELS, config.GEN_FEATURES).to(config.DEVICE)
    model_critic = Critic(config.IMG_CHANNELS, config.CRITIC_FEATURES).to(config.DEVICE)

    optimizer_gen = optim.Adam(model_gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.9))
    optimizer_critic = optim.Adam(model_critic.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.9))

    utils.load_checkpoint(f'checkpoints/generator{epoch}.pth', model_gen, optimizer_gen, config.LEARNING_RATE)
    utils.load_checkpoint(f'checkpoints/critic{epoch}.pth', model_critic, optimizer_critic, config.LEARNING_RATE)
    return model_critic, model_gen


def compute_gradient(epoch):
    disc, gen = load_models(epoch)

    fixed_noise = torch.randn(64, config.Z_DIM, 1, 1).to(config.DEVICE)
    fake = gen(fixed_noise)

    critic_fake = disc(fake)

    gradients = torch.autograd.grad(outputs=critic_fake, inputs=fake,
                                    grad_outputs=torch.ones(critic_fake.size()).cuda(),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    return gradients, fake


def visualize(epoch, etas_size=10):
    for k in range(4):
        gradient, fake = compute_gradient(epoch)

        new = []

        plt.figure()
        plt.suptitle(f'Fake image at epoch {epoch}, {k+1}/4 ', fontsize=20)
        plt.imshow(fake[0].cpu().detach().permute(1, 2, 0))
        etas = [0.001, 0.002, 0.006, 0.008, 0.01, 0.09, 0.1, 0.9]

        for eta in etas:
            # xx = np.logspace(1e-5, 1e5, etas_size)
            # eta = np.random.uniform(10 ** (-5), 0.001)
            new_image = fake - torch.mul(gradient, eta)
            new.append(new_image)

            plt.figure()
            plt.suptitle(f'eta={eta}', fontsize=20)
            plt.imshow(new_image[0].cpu().detach().permute(1, 2, 0))

        transform_PIL = transforms.ToPILImage()

        check_dirs()
        for i, news in enumerate(new):
            im = vutils.make_grid(torch.reshape(news[0], (3, 64, 64))[:64], padding=2, normalize=True)
            plt.imshow(news[0].cpu().detach().permute(1, 2, 0))
            transform_PIL(im).save(os.path.join(f'res/{k}/', str(epoch) + '_epoh_' + str(etas[i]) +
                                                "_eta_image.png"))


def check_dirs():
    for i in range(4):
        os.makedirs(f'res/{i}', exist_ok=True)
