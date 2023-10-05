import torch
from torchvision.utils import save_image
from models.generator import Generator
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

generator = Generator().to(device=device)
generator.load_state_dict(torch.load('gan_100.pth'))
generator.eval()

num_samples = 10
with torch.no_grad():
    latent_space_samples = torch.randn((num_samples, 100)).to(device=device)
    generated_samples = generator(latent_space_samples)

os.makedirs("infer_samples", exist_ok=True)
for i, generated_image in enumerate(generated_samples):
    save_image(generated_image, f'infer_samples/sample_{i}.png')

print(f"Generated {num_samples} samples and saved them.")
