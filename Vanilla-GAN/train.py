import torch
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import os
import imageio
import numpy as np
import torch.nn as nn

from data.dataset import CustomImageDataset
from models.discriminator import Discriminator
from models.generator import Generator
from utils import JSD
import config
# import matplotlib.pyplot as plt
from torchvision.transforms import v2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device != 'cuda':
    print("Not using cuda")


os.makedirs("train_samples", exist_ok=True)
os.makedirs("chk_pts", exist_ok=True)



discriminator = Discriminator().to(device=device)
generator = Generator().to(device=device)



# jsd_loss = JSD()
jsd_loss = nn.BCELoss()
optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=config.lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=config.lr)






data_dir = 'data/dataset1'

transform = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.v2.RandomHorizontalFlip(p=0.5),
    # transforms.v2.RandomVerticalFlip(p=0.5),
    # transforms.v2.AugMix(),

    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.AugMix(),

    # transforms.v2.ColorJitter(),
    # transforms.v2.GaussianBlur(p=0.5),
    transforms.Resize((64, 64)), 
    transforms.ToTensor(),
])

dataset = CustomImageDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)



_samples = next(iter(dataloader))


# for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(_samples[i].permute(1, 2, 0))
#     plt.xticks([])
#     plt.yticks([])
# plt.show()


for epoch in range(config.num_epochs):
    for n, samples_ in enumerate(dataloader):

        batch_size = len(samples_)
        samples_ = samples_.to(device=device)
        real_samples_labels = torch.ones((batch_size, 1)).to(device=device)
        latent_space_samples = torch.randn((batch_size, 100)).to(device=device)

        # print(f'Latent space samples : {latent_space_samples.shape}')
        generated_samples = generator(latent_space_samples)
        generated_samples_labels = torch.zeros((batch_size, 1)).to(device=device)


        all_samples = torch.cat((samples_, generated_samples))

        # print(f'Real samples : {samples_.shape}')  
        # print(f'Generated samples : {generated_samples.shape}')

        all_samples_labels = torch.cat(
            (real_samples_labels, generated_samples_labels)
        )

        discriminator.zero_grad()
        output_discriminator = discriminator(all_samples)
        loss_discriminator = jsd_loss(
            output_discriminator, all_samples_labels
        )
        loss_discriminator.backward()
        optimizer_discriminator.step()

        latent_space_samples = torch.randn((batch_size, 100)).to(
            device=device
        )

        generator.zero_grad()
        generated_samples = generator(latent_space_samples)
        output_discriminator_generated = discriminator(generated_samples)
        loss_generator = jsd_loss(
            output_discriminator_generated, real_samples_labels
        )
        loss_generator.backward()
        optimizer_generator.step()

        if n == batch_size - 1:
            print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
            print(f"Epoch: {epoch} Loss G.: {loss_generator}")

        if n == len(dataloader) - 1 and epoch % config.save_ == 0:
            with torch.no_grad():
                generated_samples = generator(latent_space_samples)

            for i, generated_image in enumerate(generated_samples):
                filename = f"train_samples/epoch_{epoch}_sample_{i}.png"
                generated_image_np = (generated_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                original_resolution = (generated_image.shape[1], generated_image.shape[2])
                print(original_resolution)
                imageio.imwrite(filename, generated_image_np,resolution=original_resolution)

    if epoch % config.save_ == 0:
        torch.save(generator.state_dict(), f'chk_pts/gen_{epoch}.pth')
        # torch.save(discriminator.state_dict(), f'discriminator_epoch_{epoch}.pth')