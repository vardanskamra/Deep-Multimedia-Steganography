import io
import torch
import torchvision
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import functional as TF
from PIL import Image
from torch.amp import autocast, GradScaler
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure



### Metrics:

def normalized_correlation(x, y):
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    numerator = torch.sum((x - x_mean) * (y - y_mean))
    denominator = torch.sqrt(torch.sum((x - x_mean) ** 2) * torch.sum((y - y_mean) ** 2))
    return numerator / denominator



### Transforms

transform = T.Compose([
    T.Resize((256, 256)),
    T.Lambda(lambda img: img.convert("RGB")),
    T.ToTensor(),
])



### Attacks

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



### Visualize

import matplotlib.pyplot as plt

def visualize_images(cover, secret, stego, secret_revealed):
    """
    Shows cover, secret, stego and revealed images.
    """
    
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 4, 1)
    plt.title("Cover")
    plt.imshow(cover.permute(1, 2, 0))
    plt.subplot(1, 4, 2)
    plt.title("Secret")
    plt.imshow(secret.permute(1, 2, 0))
    plt.subplot(1, 4, 3)
    plt.title("Stego")
    plt.imshow(stego.permute(1, 2, 0))
    plt.subplot(1, 4, 4)
    plt.title("Revealed")
    plt.imshow(secret_revealed.permute(1, 2, 0))
    plt.show()    

def plot_metrics(metrics):
    """
    Plots training metrics over epochs and prints final evaluation results.
    """
    
    epochs = range(1, len(metrics['loss']) + 1)
    
    # Plot Loss Curve
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 3, 1)
    plt.plot(epochs, metrics['loss'], label='Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    # Plot PSNR Curve
    plt.subplot(2, 3, 2)
    plt.plot(epochs, metrics['psnr'], label='PSNR', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR (dB)')
    plt.title('Peak Signal-to-Noise Ratio (PSNR)')
    plt.legend()

    # Plot SSIM Curve
    plt.subplot(2, 3, 3)
    plt.plot(epochs, metrics['ssim'], label='SSIM', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('SSIM')
    plt.title('Structural Similarity Index (SSIM)')
    plt.legend()

    # Plot Normalized Correlation (NC) Curve
    plt.subplot(2, 3, 4)
    plt.plot(epochs, metrics['nc'], label='Normalized Correlation', color='purple')
    plt.xlabel('Epochs')
    plt.ylabel('NC')
    plt.title('Normalized Correlation (NC)')
    plt.legend()

    # Plot Pixel Loss (Cover-Stego)
    plt.subplot(2, 3, 5)
    plt.plot(epochs, metrics['pixel_loss_cover_stego'], label='Pixel Loss (Cover-Stego)', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Pixel Loss')
    plt.title('Pixel Loss (Cover-Stego)')
    plt.legend()

    # Plot Pixel Loss (Secret-Revealed)
    plt.subplot(2, 3, 6)
    plt.plot(epochs, metrics['pixel_loss_secret_revealed'], label='Pixel Loss (Secret-Revealed)', color='brown')
    plt.xlabel('Epochs')
    plt.ylabel('Pixel Loss')
    plt.title('Pixel Loss (Secret-Revealed)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Print Final Metrics
    print("\nFinal Evaluation Metrics:")
    print(f"Final Loss: {metrics['loss'][-1]:.4f}")
    print(f"Final PSNR: {metrics['psnr'][-1]:.4f} dB")
    print(f"Final SSIM: {metrics['ssim'][-1]:.4f}")
    print(f"Final Normalized Correlation (NC): {metrics['nc'][-1]:.4f}")
    print(f"Final Pixel Loss (Cover-Stego): {metrics['pixel_loss_cover_stego'][-1]:.4f}")
    print(f"Final Pixel Loss (Secret-Revealed): {metrics['pixel_loss_secret_revealed'][-1]:.4f}\n")




### Train Loop

def train(prep_net: torch.nn.Module,
                       hide_net: torch.nn.Module,
                       reveal_net: torch.nn.Module,
                       dataloader: torch.utils.data.DataLoader,
                       optimizer: torch.optim.Optimizer,
                       beta=0.75,
                       epochs=50,
                       device=None,
                       checkpoint=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    prep_net.train()
    hide_net.train()
    reveal_net.train()
    
    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    # Hyperparameters for additional loss components:
    lambda_psnr = 0.2
    lambda_ssim = 0.2
    lambda_nc   = 0.2
    eps = 1e-8  # to avoid division by zero

    start_epoch = 0
    if checkpoint:
        prep_net.load_state_dict(checkpoint['prep_net_state_dict'])
        hide_net.load_state_dict(checkpoint['hide_net_state_dict'])
        reveal_net.load_state_dict(checkpoint['reveal_net_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        metrics = checkpoint.get('metrics', {
            'loss': [], 'psnr': [], 'ssim': [],
            'nc': [], 'pixel_loss_cover_stego': [], 'pixel_loss_secret_revealed': []
        })
    else:
        metrics = {
            'loss': [], 'psnr': [], 'ssim': [],
            'nc': [], 'pixel_loss_cover_stego': [], 'pixel_loss_secret_revealed': []
        }

    # Initialize GradScaler with the new API:
    scaler = GradScaler(device='cuda')

    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        epoch_psnr = 0.0
        epoch_ssim = 0.0
        epoch_nc = 0.0
        epoch_pixel_loss_cover_stego = 0.0
        epoch_pixel_loss_secret_revealed = 0.0
        num_batches = 0

        for i, (images, _) in enumerate(dataloader):
            if i % 2 == 0:
                cover = images.to(device)
            else:
                secret = TF.resize(images.to(device), (128, 128))
                optimizer.zero_grad()

                # Use mixed precision autocast
                with autocast(device_type='cuda'):
                    secret_prepared = prep_net(secret)
                    stego = hide_net(cover, secret_prepared)
                    # Apply a random attack during training
                    attacked_stego = random_attack(stego)
                    secret_revealed = reveal_net(attacked_stego)

                     # --- Compute partial losses ---
                    mse_cover_loss = F.mse_loss(stego, cover)  # cover reconstruction
                    mse_secret_loss = F.mse_loss(secret_revealed, secret)  # secret reconstruction
                    
                     # For cover: we want high PSNR and SSIM.
                    current_psnr = psnr(stego, cover)  # Higher is better.
                    current_ssim = ssim(stego, cover)    # In [0,1], higher is better.
                    psnr_loss = 1.0 / (current_psnr.clamp(min=eps) + eps)
                    ssim_loss = 1.0 - current_ssim.clamp(min=0.0, max=1.0)
                    cover_loss_total = mse_cover_loss + lambda_psnr * psnr_loss + lambda_ssim * ssim_loss
                    
                    # For secret: we want high normalized correlation (NC).
                    current_nc = normalized_correlation(secret.float(), secret_revealed.float())
                    nc_loss = 1.0 - current_nc.clamp(min=-1.0, max=1.0)
                    secret_loss_total = mse_secret_loss + lambda_nc * nc_loss
                
                # 1) Backprop cover-loss to Prep+Hide only 
                # Temporarily freeze Reveal net so it doesn't get gradients from cover_loss
                for param in reveal_net.parameters():
                    param.requires_grad = False
                # Backprop cover_loss
                scaler.scale(cover_loss_total).backward(retain_graph=True)
                # Unfreeze Reveal net
                for param in reveal_net.parameters():
                    param.requires_grad = True

                # 2) Backprop secret-loss to ALL networks 
                scaler.scale(beta * secret_loss_total).backward()
                # Now we can step the optimizer once
                scaler.step(optimizer)
                scaler.update()

                # For logging & metrics
                total_loss = cover_loss_total.item() + beta * secret_loss_total.item()
                
                with torch.no_grad():
                    psnr_value = psnr(stego, cover)
                    ssim_value = ssim(stego, cover)
                    nc_value = normalized_correlation(secret.float(), secret_revealed.float()) # avoids autocast for NC value
                    pixel_loss_cover_stego = F.l1_loss(cover, stego)
                    pixel_loss_secret_revealed = F.l1_loss(secret, secret_revealed)

                epoch_loss += total_loss
                epoch_psnr += psnr_value.item()
                epoch_ssim += ssim_value.item()
                epoch_nc += nc_value.item()
                epoch_pixel_loss_cover_stego += pixel_loss_cover_stego.item()
                epoch_pixel_loss_secret_revealed += pixel_loss_secret_revealed.item()
                num_batches += 1

        metrics['loss'].append(epoch_loss / num_batches)
        metrics['psnr'].append(epoch_psnr / num_batches)
        metrics['ssim'].append(epoch_ssim / num_batches)
        metrics['nc'].append(epoch_nc / num_batches)
        metrics['pixel_loss_cover_stego'].append(epoch_pixel_loss_cover_stego / num_batches)
        metrics['pixel_loss_secret_revealed'].append(epoch_pixel_loss_secret_revealed / num_batches)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {metrics['loss'][-1]:.4f}, PSNR: {metrics['psnr'][-1]:.4f}, "
              f"SSIM: {metrics['ssim'][-1]:.4f}, NC: {metrics['nc'][-1]:.4f}, "
              f"Pixel Loss (Cover-Stego): {metrics['pixel_loss_cover_stego'][-1]:.4f}, "
              f"Pixel Loss (Secret-Revealed): {metrics['pixel_loss_secret_revealed'][-1]:.4f}")

        checkpoint = {
            'epoch': epoch + 1,
            'prep_net_state_dict': prep_net.state_dict(),
            'hide_net_state_dict': hide_net.state_dict(),
            'reveal_net_state_dict': reveal_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        torch.save(checkpoint, "model_checkpoint.pth")

    return metrics




### Test Loop

def test(prep_net: nn.Module,
                      hide_net: nn.Module,
                      reveal_net: nn.Module,
                      dataloader: torch.utils.data.DataLoader,
                      beta=0.75,
                      attacks=None,
                      visualize=5, # Visualize after every 5 batches
                      device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    prep_net.eval()
    hide_net.eval()
    reveal_net.eval()
    
    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    # Hyperparameters for additional loss components:
    lambda_psnr = 0.2
    lambda_ssim = 0.2
    lambda_nc   = 0.2
    eps = 1e-8  # to avoid division by zero
    
    metrics = {
        'loss': [], 'psnr': [], 'ssim': [],
        'nc': [], 'pixel_loss_cover_stego': [], 'pixel_loss_secret_revealed': []
    }
    
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_nc = 0.0
    total_pixel_loss_cover_stego = 0.0
    total_pixel_loss_secret_revealed = 0.0
    num_batches = 0


    with torch.inference_mode():
        for i, (images, _) in enumerate(dataloader):
            if i % 2 == 0:
                cover = images.to(device)
            else:
                secret = TF.resize(images.to(device), (128, 128))
                secret_prepared = prep_net(secret)
                stego = hide_net(cover, secret_prepared)
                # Apply the specified attack(s) (can be a single attack or a list)
                attacked_stego = apply_attacks(stego, attacks)
                secret_revealed = reveal_net(attacked_stego)

                mse_cover_loss = F.mse_loss(stego, cover)  
                mse_secret_loss = F.mse_loss(secret_revealed, secret)
                
                current_psnr = psnr(stego, cover)  
                current_ssim = ssim(stego, cover)    
                psnr_loss = 1.0 / (current_psnr.clamp(min=eps) + eps)
                ssim_loss = 1.0 - current_ssim.clamp(min=0.0, max=1.0)
                cover_loss_total = mse_cover_loss + lambda_psnr * psnr_loss + lambda_ssim * ssim_loss
                
                current_nc = normalized_correlation(secret.float(), secret_revealed.float())
                nc_loss = 1.0 - current_nc.clamp(min=-1.0, max=1.0)
                secret_loss_total = mse_secret_loss + lambda_nc * nc_loss
                
                loss = cover_loss_total.item() + beta * secret_loss_total.item()
                psnr_value = psnr(stego, cover)
                ssim_value = ssim(stego, cover)
                nc_value = normalized_correlation(secret, secret_revealed)
                pixel_loss_cover_stego = F.l1_loss(cover, stego)
                pixel_loss_secret_revealed = F.l1_loss(secret, secret_revealed)

                total_loss += loss
                total_psnr += psnr_value.item()
                total_ssim += ssim_value.item()
                total_nc += nc_value.item()
                total_pixel_loss_cover_stego += pixel_loss_cover_stego.item()
                total_pixel_loss_secret_revealed += pixel_loss_secret_revealed.item()
                num_batches += 1

                if visualize and (num_batches % visualize == 0):
                    visualize_images(cover[0].cpu(), secret[0].cpu(),
                                     stego[0].squeeze(0).cpu(), secret_revealed[0].squeeze(0).cpu())
    
    metrics['loss'].append(total_loss / num_batches)
    metrics['psnr'].append(total_psnr / num_batches)
    metrics['ssim'].append(total_ssim / num_batches)
    metrics['nc'].append(total_nc / num_batches)
    metrics['pixel_loss_cover_stego'].append(total_pixel_loss_cover_stego / num_batches)
    metrics['pixel_loss_secret_revealed'].append(total_pixel_loss_secret_revealed / num_batches)

    print(f"\nTest Results (Attacks: {attacks}): Loss: {metrics['loss'][-1]:.4f}, "
          f"PSNR: {metrics['psnr'][-1]:.4f}, SSIM: {metrics['ssim'][-1]:.4f}, NC: {metrics['nc'][-1]:.4f}, "
          f"Pixel Loss (Cover-Stego): {metrics['pixel_loss_cover_stego'][-1]:.4f}, "
          f"Pixel Loss (Secret-Revealed): {metrics['pixel_loss_secret_revealed'][-1]:.4f}\n")

    return metrics



"""
def run():
    
    import os
    from torchvision import datasets
    from torch.utils.data import DataLoader
    
    
    try:
        import torchmetrics
    except ImportError:
        import subprocess
        subprocess.check_call(["pip", "install", "torchmetrics"])
        import torchmetrics
        
        
    try:
        from torchinfo import summary
    except ImportError:
        import subprocess
        subprocess.check_call(["pip", "install", "torchinfo"])
        from torchinfo import summary
        
        
    repo_url = "https://raw.githubusercontent.com/vardanskamra/Deep-Multimedia-Steganography/main"
    file = "helper_functions.py"
    file_url = f"{repo_url}/{file}"
    os.system(f"curl -s -f {file_url} -o {file}")
    print("Downloaded all file successfully!")
    
    
    from helper_functions import (loss_function, 
    train, 
    test,
    random_attack, 
    apply_attacks,
    plot_metrics,
    visualize_images)
    from helper_functions import transform
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    dataset = datasets.Caltech256(root='./data', download=True, transform=transform)
    subset_size = int(1 * len(dataset))  
    train_size = int(0.8 * subset_size)  
    test_size = subset_size - train_size  
    subset_dataset, _ = torch.utils.data.random_split(dataset, [subset_size, len(dataset) - subset_size])
    train_dataset, test_dataset = torch.utils.data.random_split(subset_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=os.cpu_count(), drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=True)
    
    
    print(f"Train Dataset Length: {len(train_dataset)}")
    print(f"Test Dataset Length: {len(test_dataset)}")
    print(f"Train DataLoader Length: {len(train_loader)}")
    print(f"Test DataLoader Length: {len(test_loader)}")
    sample, label = next(iter(train_loader))
    print(f"Sample Shape: {sample.shape}")
    print(f"Label Shape: {label.shape}")
    
    
    
    
    
    print(f"Device: {device}")
    prep_net = nn.DataParallel(PrepNetwork()).to(device)
    hide_net = nn.DataParallel(HidingNetwork()).to(device)
    reveal_net = nn.DataParallel(RevealNetwork()).to(device)
        
        
    print("PrepNetwork Summary:")
    print(summary(prep_net, input_size=(32, 3, 256, 256)))
    cover = torch.randn(32, 3, 256, 256).to(device)
    secret_prepared = torch.randn(32, 3, 256, 256).to(device)
    print("\nHidingNetwork Summary:")
    print(summary(hide_net, input_data=[cover, secret_prepared])) # 6 channels (cover + secret_prepared)
    print("\nRevealNetwork Summary:")
    print(summary(reveal_net, input_size=(32, 3, 256, 256)))
    
    
    optimizer = torch.optim.Adam(list(prep_net.parameters()) +
                       list(hide_net.parameters()) +
                       list(reveal_net.parameters()), lr=0.001)
    
    
    train_metrics = train(dataloader=train_loader,
                prep_net=prep_net,
                hide_net=hide_net,
                reveal_net=reveal_net,
                optimizer=optimizer,
                loss_fn=loss_function,
                beta=1.25,
                epochs=5,
                checkpoint=None,
                device=device)
    plot_metrics(train_metrics)
    test_metrics = test(prep_net=prep_net,
               hide_net=hide_net,
               reveal_net=reveal_net,
               dataloader=test_loader,
               loss_fn = loss_function,
               beta = 1.25,
               visualize = 2,
               attacks=None,                  
               device=device)
    torch.save(prep_net.state_dict(), "/kaggle/working/prep_net.pth")
    torch.save(hide_net.state_dict(), "/kaggle/working/hide_net.pth")
    torch.save(reveal_net.state_dict(), "/kaggle/working/reveal_net.pth")
"""