import os
from torch.utils.data import Dataset, DataLoader
import imageio

class CustomImageDataset(Dataset):
    def __init__(self, path, transform):
        self.paths = path
        self.transform = transform
        files = os.listdir(path)
        self.imgs_ = [os.path.join(path,file) for file in files]
        
    def __len__(self):
        return len(self.paths)
    

    def __getitem__(self, index):
        image_path = self.imgs_[index]
        image = imageio.imread(image_path,pilmode="RGB")
        
        if self.transform:
            image_tensor = self.transform(image)
            
        return image_tensor
    

