import torch
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from utils.metrics import loss_function
from utils.metrics import normalized_correlation

def train(prep_net: torch.nn.Module,
          hide_net: torch.nn.Module,
          reveal_net: torch.nn.Module,
          dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn = loss_function,
          beta = 0.75,
          epochs = 50,
          device = None):
  
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    prep_net.train(); hide_net.train(); reveal_net.train()

    # Initialize metric calculators
    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # Dictionary to store metrics for each epoch
    metrics = {
        'loss': [],
        'psnr': [],
        'ssim': [],
        'nc': [],
        'pixel_loss_cover_stego': [],
        'pixel_loss_secret_revealed': []
    }

    for epoch in range(epochs):

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
                secret = images.to(device)
                optimizer.zero_grad()
                secret_prepared = prep_net(secret)
                stego = hide_net(cover, secret_prepared)
                secret_revealed = reveal_net(stego)
                
                # Compute loss
                loss = loss_fn(cover, stego, secret, secret_revealed, beta=beta)
                loss.backward()
                optimizer.step()
                
                # Compute metrics
                psnr_value = psnr(stego, cover)
                ssim_value = ssim(stego, cover)
                nc_value = normalized_correlation(secret, secret_revealed)
                pixel_loss_cover_stego = torch.nn.functional.l1_loss(cover, stego)
                pixel_loss_secret_revealed = torch.nn.functional.l1_loss(secret, secret_revealed)
                
                # Accumulate metrics for the epoch
                epoch_loss += loss.item()
                epoch_psnr += psnr_value.item()
                epoch_ssim += ssim_value.item()
                epoch_nc += nc_value.item()
                epoch_pixel_loss_cover_stego += pixel_loss_cover_stego.item()
                epoch_pixel_loss_secret_revealed += pixel_loss_secret_revealed.item()
                num_batches += 1

        # Average metrics over the epoch
        metrics['loss'].append(epoch_loss / num_batches)
        metrics['psnr'].append(epoch_psnr / num_batches)
        metrics['ssim'].append(epoch_ssim / num_batches)
        metrics['nc'].append(epoch_nc / num_batches)
        metrics['pixel_loss_cover_stego'].append(epoch_pixel_loss_cover_stego / num_batches)
        metrics['pixel_loss_secret_revealed'].append(epoch_pixel_loss_secret_revealed / num_batches)

        # Print metrics for the epoch
        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Loss: {metrics['loss'][-1]:.4f}, "
              f"PSNR: {metrics['psnr'][-1]:.4f}, "
              f"SSIM: {metrics['ssim'][-1]:.4f}, "
              f"NC: {metrics['nc'][-1]:.4f}, "
              f"Pixel Loss (Cover-Stego): {metrics['pixel_loss_cover_stego'][-1]:.4f}, "
              f"Pixel Loss (Secret-Revealed): {metrics['pixel_loss_secret_revealed'][-1]:.4f}")

    return metrics