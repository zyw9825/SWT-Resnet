import cv2
from pyWavelet import pywt_swt2
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset
from SWTResNet50 import SWTResNet
import albumentations as A
from torch.utils.data import DataLoader
from fit import fit
import random
DEVICE = "cuda:0"


class ImageFolder(Dataset):
    """
    Custom dataset class for loading images and their corresponding labels.
    This class applies wavelet transformations and optional augmentations to images.

    Parameters:
    - train_dir: Directory containing training data.
    - target_dir: Directory containing target labels.
    - transform: Data augmentations to apply to images.
    - train: Boolean indicating whether it's training data or not.
    """

    def __init__(self, train_dir=None, target_dir=None, transform=None, train=True):
        self.transform = transform
        self.data = []
        self.targets = []
        self.class_data = {}
        self.train = train
        self.data = np.load(train_dir)  # Load training data
        self.targets = np.load(target_dir)  # Load target labels
        self.all_data = []
        for i in range(len(self.data)):
            if self.transform:
                img = cv2.merge(self.data[i])  # Merge BGR channels
                augmentations = self.transform(image=img)  # Apply data augmentations
                img = augmentations["image"]
                img = self.wavelet_transform(img)  # Apply wavelet transform
                self.all_data.append((img, self.targets[i]))



    def wavelet_transform(self, img):
        """
        Apply wavelet transform to the image by splitting into B, G, R channels.
        """
        B, G, R = cv2.split(img)
        return pywt_swt2(np.array([B, G, R]), 4, "rbio3.5")  # Perform wavelet transformation

    def __len__(self):
        """
        Return the length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Retrieve an image and its corresponding target label by index, apply transformations and wavelet.
        """
        # img, target = self.data[index], self.targets[index]
        img, target = self.all_data[index]

        img = torch.tensor(img, dtype=torch.float32).view(5, 3, IMAGE_SIZE, IMAGE_SIZE).cuda()  # Convert to tensor
        img = [tensor for tensor in img]
        # if self.transform:
        #     img = cv2.merge(img)  # Merge BGR channels
        #     augmentations = self.transform(image=img)  # Apply data augmentations
        #     img = augmentations["image"]
        #     img = self.wavelet_transform(img)  # Apply wavelet transform
        #     img = torch.tensor(img, dtype=torch.float32).view(5, 3, IMAGE_SIZE, IMAGE_SIZE).cuda()  # Convert to tensor
        #     img = [tensor for tensor in img]
        target = torch.from_numpy(np.array(target, dtype=np.int64))  # Convert target to tensor
        return img, target


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    NUM_CLASSES = 6  # Output dimension (number of classes)
    GRAYSCALE = False  # If images are grayscale or color

    model = SWTResNet(num_classes=NUM_CLASSES, grayscale=GRAYSCALE)  # Initialize model
    params = torch.load("model_epoch_47.pth")
    model.load_state_dict(params)
    model.cuda()

    IMAGE_SIZE = 224  # Define image size

    # Define random values for parameters
    random_rotation_limit = random.uniform(5, 10)  # Random rotation between 5 and 15 degrees
    random_scale = random.uniform(0.8, 1.0)  # Random scale between 0.8 and 1.0
    random_var_limit = (random.uniform(1.0, 5.0), random.uniform(10.0, 20.0))  # Random Gaussian noise variance limits

    # Define the augmentation pipeline
    train_transform = A.Compose([
        A.Resize(width=IMAGE_SIZE, height=IMAGE_SIZE),  # Resize image
        A.Rotate(limit=random_rotation_limit, p=0.5),  # Random rotation with random limit
        A.RandomResizedCrop(height=224, width=224, scale=(random_scale, 1.0), p=1),  # Random scale for cropping
        A.HorizontalFlip(p=0.5),  # Random horizontal flip
        A.GaussNoise(var_limit=random_var_limit, p=0.5),  # Random Gaussian noise
    ])

    # Define data augmentations for testing
    test_transform = A.Compose([
        A.Resize(width=IMAGE_SIZE, height=IMAGE_SIZE),
    ])

    BATCH_SIZE = 32  # Define batch size
    epochs = 50  # Define number of epochs

    # Load training data
    train_data_simple = ImageFolder(train_dir="AMT_trainval_x_200.npy",
                                    target_dir="AMT_trainval_y_200.npy",
                                    transform=train_transform,
                                    train=True)

    # Load testing data
    test_data_simple = ImageFolder(train_dir="AMT_test_x_200.npy",
                                   target_dir="AMT_test_y_200.npy",
                                   transform=test_transform,
                                   train=False)

    # Create data loaders
    train_loader = DataLoader(train_data_simple,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=0)

    val_loader = DataLoader(test_data_simple,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            num_workers=0)

    # Set up optimizer, loss function, and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, 8, gamma=0.1)

    # Train the model
    fit(train_loader, val_loader, model, criterion, optimizer, scheduler, epochs)


