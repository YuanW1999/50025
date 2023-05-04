import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision.datasets import ImageFolder
transform = transforms.Compose([
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
path="/home/featurize/data/ma2banma/trainA"
trainA_dataset = ImageFolder(root=path, transform=transform)
trainA_loader = DataLoader(trainA_dataset, batch_size=1, shuffle=True)

path="/home/featurize/data/ma2banma/trainB"
trainB_dataset = ImageFolder(root=path, transform=transform)
trainB_loader = DataLoader(trainB_dataset, batch_size=1, shuffle=True)

path="/home/featurize/data/ma2banma/testA"
testA_dataset = ImageFolder(root=path, transform=transform)
testA_loader = DataLoader(testA_dataset, batch_size=1, shuffle=True)

path="/home/featurize/data/ma2banma/testB"
testB_dataset = ImageFolder(root=path, transform=transform)
testB_loader = DataLoader(testB_dataset, batch_size=1, shuffle=True)
