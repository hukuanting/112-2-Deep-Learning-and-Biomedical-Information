# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# from PIL import Image

# # Function to binarize and save images
# def save_images(images, folder, threshold=0.5):
#     if not os.path.exists(folder):
#         os.makedirs(folder)
#     for i, img in enumerate(images):
#         # Reshape and process the image correctly
#         img = img.transpose(1, 2, 0)  # Change shape from (3, 64, 64) to (64, 64, 3)
#         img = (img + 1) / 2.0  # Rescale to [0, 1]
#         img = (img > threshold).astype(np.uint8) * 255  # Binarize image
#         img = Image.fromarray(img)
#         img.save(os.path.join(folder, f'image_{i}.png'))

# flag_gpu = 1
# device = 'cuda:0' if (torch.cuda.is_available() & flag_gpu) else 'cpu'
# print(device)

# G = torch.load(r'C:\Users\USER\Desktop\Lab\Code\DCGAN\DCGAN_Generator_1.pth', map_location=device)
# G.eval()

# latent_dim = 100

# # Generate and display images (Experiment 1)
# noise = torch.randn(20, latent_dim, 1, 1).to(device)
# fake_inputs = G(noise)
# print(fake_inputs.shape)  # Print the shape of generated images
# imgs_numpy = fake_inputs.data.cpu().numpy()
# save_images(imgs_numpy[:16], folder=r'C:\Users\USER\Desktop\Lab\Code\DCGAN\generate_images')


import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

# Function to binarize and save images
def save_images(images, folder, threshold=0.5):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for i, img in enumerate(images):
        # Reshape and process the image correctly for grayscale
        img = img.squeeze(0)  # Remove the channel dimension (shape: (64, 64))
        img = (img + 1) / 2.0  # Rescale to [0, 1]
        img = (img > threshold).astype(np.uint8) * 255  # Binarize image
        img = Image.fromarray(img, mode='L')  # Use mode 'L' for grayscale images
        img.save(os.path.join(folder, f'image_{i}.png'))

flag_gpu = 1
device = 'cuda:0' if (torch.cuda.is_available() & flag_gpu) else 'cpu'
print(device)

G = torch.load(r'C:\Users\USER\Desktop\Lab\Code\DCGAN\DCGAN_Generator_1.pth', map_location=device)
G.eval()

latent_dim = 100

# Generate and display images (Experiment 1)
noise = torch.randn(20, latent_dim, 1, 1).to(device)
fake_inputs = G(noise)
print(fake_inputs.shape)  # Print the shape of generated images
imgs_numpy = fake_inputs.data.cpu().numpy()
save_images(imgs_numpy[:16], folder=r'C:\Users\USER\Desktop\Lab\Code\DCGAN\generate_images')

