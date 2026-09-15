import random
import torch
from torchvision.transforms import functional as F

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            _, height, width = image.shape  # image is [C, H, W] after ToTensor
            image = image.flip(-1)  # Flip horizontally
            
            if 'boxes' in target:
                bbox = target['boxes']
                # Flip box coordinates
                bbox = bbox[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([width, 0, width, 0])
                target['boxes'] = bbox
        return image, target