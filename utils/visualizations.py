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
