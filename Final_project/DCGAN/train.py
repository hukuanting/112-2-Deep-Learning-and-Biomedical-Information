import torch
import os
from torch.utils.data import DataLoader
import numpy as np
from torch import nn
from torch import optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt
 
from dataset import Custom_dataset
from model import Generator, Discriminator, weights_init
 
 
 
batch_size = 128
# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64
# Number of channels in the training images. For color images this is 3
nc = 1
# Size of z latent vector (i.e. size of generator input)
nz = 100
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64
# Number of training epochs
num_epochs = 100
# Learning rate for optimizers
lr = 0.0002
# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
 
# 1. load real image
root = r"C:\Users\USER\Desktop\class\train\0"
dataset = Custom_dataset(root)
print(len(dataset))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print(len(dataloader))
 
 
# 2. load generator and discriminator
netG = torch.load(r'C:\Users\USER\Desktop\Lab\Code\DCGAN\DCGAN_Generator_1.pth', map_location=device)
netG.train()
# netG.load_state_dict(torch.load(r'DCGAN\DCGAN_Generator_1.pth'))


# if (device.type == 'cuda') and (ngpu > 1):
#     netG = nn.DataParallel(netG, list(range(ngpu)))
# netG.apply(weights_init)   #  to ``mean=0``, ``stdev=0.02``.
# print(netG)
 
 
# Create the Discriminator
netD = torch.load(r'C:\Users\USER\Desktop\Lab\Code\DCGAN\DCGAN_Discriminator_1.pth', map_location=device)
netD.train()
# netD.load_state_dict(torch.load(r'DCGAN\DCGAN_Discriminator_1.pth'))


# if (device.type == 'cuda') and (ngpu > 1):
#     netD = nn.DataParallel(netD, list(range(ngpu)))
# netD.apply(weights_init)   # `to mean=0, stdev=0.2``.
# print(netD)
 
# 3. loss function
criterion = nn.BCELoss()
 
# 4. Create batch of latent vectors that we will use to visualize
fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)
 
real_label = 1.
fake_label = 0.
 
# 5. Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
 
 
# 6. train loop
img_list = []
G_losses = []
D_losses = []
iters = 0
 
print("Starting Training Loop...")
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        real_cpu = data.to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        output = netD(real_cpu).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()
 
        ## Train with all-fake batch
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach()).view(-1)  # .detach表示不迭代梯度，此时netG不动
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
 
        errD = errD_real + errD_fake
        optimizerD.step()
 
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()
 
        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
 
        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())
 
        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
 
                img_batch = vutils.make_grid(fake[:64], padding=2, normalize=False)
                img_batch = np.transpose(img_batch, (1,2,0)).numpy()
                img_batch = img_batch[:,:,:] * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
 
                plt.figure(figsize=(8,8))
                plt.axis("off")
                plt.title("Generate Images")
                plt.imshow(img_batch)
                model_path = "result/"+str(iters)
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                plt.savefig(model_path+"/fake.png")
                torch.save(netG.state_dict(), model_path+"/model.pth")
            # img_list.append(vutils.make_grid(fake[:64], padding=2, normalize=True))
 
        iters += 1
 
print("End Training Loop...")

x = range(iters)
plt.figure(figsize=(10,5))
plt.title("Loss")
plt.xlabel("iters")
plt.ylabel("loss value")
plt.plot(x, D_losses,'-',label="D_loss", color='r')
plt.plot(x, G_losses,'-',label="G_loss", color='b')
plt.legend()
plt.grid(True)
plt.savefig("loss.png")


torch.save(netG, r'DCGAN\DCGAN_Generator_1.pth')
torch.save(netD, r'DCGAN\DCGAN_Discriminator_1.pth')
print('Model saved.')