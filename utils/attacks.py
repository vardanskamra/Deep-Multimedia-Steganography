import torch
import random
import numpy as np
from torchvision.transforms import functional as TF
import torchvision.transforms as T
import io
from PIL import Image

def attack_gaussian(image, sigma=1.0):
    """Applies Gaussian blur with a given sigma value."""
    return TF.gaussian_blur(image, kernel_size=5, sigma=sigma)

def attack_salt_and_pepper(image, amount=0.05, s_vs_p=0.5):
    """Applies salt-and-pepper noise by modifying the tensor directly."""
    image_np = image.detach().cpu().numpy()  # Detach to prevent autograd issues
    noise = np.random.rand(*image_np.shape)
    image_np[noise < amount * s_vs_p] = 0.0
    image_np[noise > 1 - amount * (1 - s_vs_p)] = 1.0
    return torch.from_numpy(image_np).to(image.device)

def attack_jpeg(image, quality=50):
    """Applies JPEG compression by converting to PIL, saving, and reloading."""
    if image.dim() == 4:  # Handle batch processing
        return torch.stack([attack_jpeg(img, quality) for img in image])

    buffer = io.BytesIO()
    image_pil = TF.to_pil_image(image.detach().cpu().squeeze(0))
    image_pil.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)

    image_reloaded = Image.open(buffer)
    return TF.to_tensor(image_reloaded).to(image.device)

def random_attack(image):
    """Randomly selects an attack including three Gaussian variations or no attack."""
    attacks = ['gaussian_1', 'gaussian_0.1', 'gaussian_0.05', 'salt_and_pepper', 'none']
    chosen = random.choice(attacks)

    if chosen == 'gaussian_1':
        return attack_gaussian(image, sigma=1.0)
    elif chosen == 'gaussian_0.1':
        return attack_gaussian(image, sigma=0.1)
    elif chosen == 'gaussian_0.05':
        return attack_gaussian(image, sigma=0.05)
    elif chosen == 'salt_and_pepper':
        return attack_salt_and_pepper(image)
    
    return image  # No attack

def apply_attacks(image, attacks):
    """Applies a list of attacks sequentially."""
    if attacks is None:
        return image
    if isinstance(attacks, str):
        attacks = [attacks]
    
    for attack in attacks:
        if attack == 'gaussian_1':
            image = attack_gaussian(image, sigma=1.0)
        elif attack == 'gaussian_0.1':
            image = attack_gaussian(image, sigma=0.1)
        elif attack == 'gaussian_0.05':
            image = attack_gaussian(image, sigma=0.05)
        elif attack == 'salt_and_pepper':
            image = attack_salt_and_pepper(image)
        elif attack == 'jpeg':
            image = attack_jpeg(image)
    
    return image
