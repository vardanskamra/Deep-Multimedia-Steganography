import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from utils.metrics import loss_function
from utils.metrics import normalized_correlation
from utils.visualizations import visualize_images
from utils.attacks import *

def test(prep_net: torch.nn.Module,
         hide_net: torch.nn.Module,
         reveal_net: torch.nn.Module,
         dataloader: torch.utils.data.DataLoader,
         loss_fn = loss_function,
         beta = 0.75,
         visualize = True,
         device=None):
    
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Put models in evaluation mode
    prep_net.eval()
    hide_net.eval()
    reveal_net.eval()

    # Initialize metric calculators
    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # Dictionary to store metrics
    metrics = {
        'loss': [],
        'psnr': [],
        'ssim': [],
        'nc': [],
        'pixel_loss_cover_stego': [],
        'pixel_loss_secret_revealed': []
    }

    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_nc = 0.0
    total_pixel_loss_cover_stego = 0.0
    total_pixel_loss_secret_revealed = 0.0
    num_batches = 0

    with torch.inference_mode():  # Ensures gradients are not computed (memory efficient)
        for i, (images, _) in enumerate(dataloader):
            if i % 2 == 0:
                cover = images.to(device)
            else:
                secret = images.to(device)

                secret_prepared = prep_net(secret)
                stego = hide_net(cover, secret_prepared)
                secret_revealed = reveal_net(stego)

                # Compute loss
                loss = loss_fn(cover, stego, secret, secret_revealed, beta=beta)

                # Compute metrics
                psnr_value = psnr(stego, cover)
                ssim_value = ssim(stego, cover)
                nc_value = normalized_correlation(secret, secret_revealed)
                pixel_loss_cover_stego = torch.nn.functional.l1_loss(cover, stego)
                pixel_loss_secret_revealed = torch.nn.functional.l1_loss(secret, secret_revealed)

                # Accumulate metrics
                total_loss += loss.item()
                total_psnr += psnr_value.item()
                total_ssim += ssim_value.item()
                total_nc += nc_value.item()
                total_pixel_loss_cover_stego += pixel_loss_cover_stego.item()
                total_pixel_loss_secret_revealed += pixel_loss_secret_revealed.item()
                num_batches += 1
            
                if (visualize == True) and (num_batches % 10 == 0):
                    visualize_images(cover[0].cpu(), secret[0].cpu(), stego[0].squeeze(0).cpu(), secret_revealed[0].squeeze(0).cpu())

    # Average metrics over the dataset
    metrics['loss'].append(total_loss / num_batches)
    metrics['psnr'].append(total_psnr / num_batches)
    metrics['ssim'].append(total_ssim / num_batches)
    metrics['nc'].append(total_nc / num_batches)
    metrics['pixel_loss_cover_stego'].append(total_pixel_loss_cover_stego / num_batches)
    metrics['pixel_loss_secret_revealed'].append(total_pixel_loss_secret_revealed / num_batches)

    # Print final evaluation results
    print(f"\nTest Results: "
          f"Loss: {metrics['loss'][-1]:.4f}, "
          f"PSNR: {metrics['psnr'][-1]:.4f}, "
          f"SSIM: {metrics['ssim'][-1]:.4f}, "
          f"NC: {metrics['nc'][-1]:.4f}, "
          f"Pixel Loss (Cover-Stego): {metrics['pixel_loss_cover_stego'][-1]:.4f}, "
          f"Pixel Loss (Secret-Revealed): {metrics['pixel_loss_secret_revealed'][-1]:.4f}\n")

    return metrics



def test_with_attacks(prep_net: nn.Module,
                      hide_net: nn.Module,
                      reveal_net: nn.Module,
                      dataloader: torch.utils.data.DataLoader,
                      loss_fn,
                      beta=0.75,
                      attacks=None,
                      visualize=5, # Visualize after every 5 batches
                      device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    prep_net.eval()
    hide_net.eval()
    reveal_net.eval()
    
    from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
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

    from utils.metrics import normalized_correlation  # assumed available
    from utils.visualizations import visualize_images  # assumed available

    with torch.inference_mode():
        for i, (images, _) in enumerate(dataloader):
            if i % 2 == 0:
                cover = images.to(device)
            else:
                secret = images.to(device)
                secret_prepared = prep_net(secret)
                stego = hide_net(cover, secret_prepared)
                # Apply the specified attack(s) (can be a single attack or a list)
                attacked_stego = apply_attacks(stego, attacks)
                secret_revealed = reveal_net(attacked_stego)

                loss = loss_fn(cover, stego, secret, secret_revealed, beta=beta)
                psnr_value = psnr(stego, cover)
                ssim_value = ssim(stego, cover)
                nc_value = normalized_correlation(secret, secret_revealed)
                pixel_loss_cover_stego = F.l1_loss(cover, stego)
                pixel_loss_secret_revealed = F.l1_loss(secret, secret_revealed)

                total_loss += loss.item()
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