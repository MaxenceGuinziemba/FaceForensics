from torchvision import transforms
from src.data.augmentations import JPEGCompression, GaussianNoise, CutOut

IMAGE_SIZE = 380
NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD = [0.5, 0.5, 0.5]

xception_default_data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomGrayscale(p=0.05),
        JPEGCompression(quality_range=(60, 95), p=0.3),
        transforms.ToTensor(),
        GaussianNoise(std_range=(0.005, 0.015), p=0.2),
        CutOut(n_holes=1, length=20, p=0.1),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ]),
    'val': transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ]),
    'test': transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ]),
}
