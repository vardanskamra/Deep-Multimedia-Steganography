import torch
import torchvision.transforms as transforms

train_test_transform = transforms.Compose([
    transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float32)
])

inference_transform = transforms.Compose([
    transforms.Resize((128, 128))
])

caltech256_train_test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))   # Normalize to [-1, 1]
])

caltech256_train_test_transform_256x256 = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))   # Normalize to [-1, 1]
])