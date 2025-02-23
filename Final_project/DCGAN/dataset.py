from torch.utils.data import DataLoader,Dataset
from torchvision import transforms as T
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
import torch
import torchvision.utils as vutils
 
 
class Custom_dataset(Dataset):
    def __init__(self, root, transforms=None):
        imgs = []
        for path in os.listdir(root):
            imgs.append(os.path.join(root, path))
 
        self.imgs = imgs
        if transforms is None:
            # normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            normalize = T.Normalize(mean=[0.5], std=[0.5])
 
            self.transforms = T.Compose([
                    T.Resize(64),
                    T.CenterCrop(64),
                    T.ToTensor(),
                    normalize
            ])
        else:
            self.transforms = transforms
             
    def __getitem__(self, index):
        img_path = self.imgs[index]
 
        data = Image.open(img_path)
        if data.mode != "L":
            data = data.convert("L")
        data = self.transforms(data)
        return data
 
    def __len__(self):
        return len(self.imgs)
 
 
if __name__ == "__main__":
    root = r"C:\Users\USER\Desktop\class\train\0"
    train_dataset = Custom_dataset(root)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
    real_batch = next(iter(train_dataloader))
    real_batch = np.transpose(vutils.make_grid(real_batch.to(device)[:64], padding=2, normalize=False).cpu(),(1,2,0)).numpy()
    real_batch = real_batch[:,:,:] * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(real_batch)
    plt.show()