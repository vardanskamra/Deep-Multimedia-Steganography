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

simple_train_test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float32)
])