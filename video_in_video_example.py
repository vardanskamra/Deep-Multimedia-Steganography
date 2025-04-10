import torch
import cv2
import os

from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.transforms import ToPILImage
from utils.metrics import loss_function
from utils.metrics import normalized_correlation
from utils.visualizations import visualize_images
from utils.inference import remove_module_prefix
from models.model_9 import PrepNetwork, HidingNetwork, RevealNetwork
from tqdm import tqdm



class VideoFramesDataset(Dataset):
    def __init__(self, frames_folder_cover, frames_folder_secret, transform1, transform2):
        self.frames_folder_cover = frames_folder_cover
        self.frames_folder_secret = frames_folder_secret
        self.transform1 = transform1
        self.transform2 = transform2
        self.frame_files_cover = sorted(os.listdir(frames_folder_cover))  # Ensure order is maintained
        self.frame_files_secret = sorted(os.listdir(frames_folder_secret))  # Ensure order is maintained
    
    def __len__(self):
        return min(len(self.frame_files_cover), len(self.frame_files_secret)) 
    
    def __getitem__(self, idx):
        img_path_cover = os.path.join(self.frames_folder_cover, self.frame_files_cover[idx])
        img_path_secret = os.path.join(self.frames_folder_secret, self.frame_files_secret[idx])
        cover = Image.open(img_path_cover).convert("RGB")
        secret = Image.open(img_path_secret).convert("RGB")
        cover = self.transform1(cover)   # 256x256
        secret = self.transform2(secret) # 128x128
        return cover, secret


def extract_frames(video_path, output_folder="video_processing/frames"):
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Break if no more frames are available
        
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
    
    cap.release()
    print(f"Extracted {frame_count} frames to '{output_folder}' folder.")



def process_video(prep_net: torch.nn.Module,
         hide_net: torch.nn.Module,
         reveal_net: torch.nn.Module,
         dataloader: torch.utils.data.DataLoader,
         secret: torch.Tensor,
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
    
    # Folder to store stego frames
    stego_folder = "video_processing/stego_frames"
    if not os.path.exists(stego_folder):
        os.makedirs(stego_folder)

    # Tensor to Image Transform
    to_pil = ToPILImage()
    
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
    frame_index = 0
    
    with torch.inference_mode():  
        for cover, secret in tqdm(dataloader):
            cover = cover.to(device)
            secret = secret.to(device)
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
            
            # Save each frame in order
            for i in range(stego.shape[0]):  # Iterate over batch
                pil_img = to_pil(stego[i].cpu())  # Convert tensor to PIL image
                frame_path = os.path.join(stego_folder, f"stego_{frame_index:04d}.png")
                pil_img.save(frame_path)  # Save frame
                frame_index += 1  # Increment frame counter
        
            if (visualize == True) and (num_batches % 5 == 0):
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


def create_video_from_frames(frames_folder="video_processing/stego_frames", output_video="video_processing/output_video.mp4", fps=30):
    # Get the list of frames
    frame_files = sorted(os.listdir(frames_folder))  # Ensure correct order
    if not frame_files:
        print("Error: No frames found in the folder.")
        return

    # Read the first frame to get video dimensions
    first_frame = cv2.imread(os.path.join(frames_folder, frame_files[0]))
    height, width, _ = first_frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Write frames to the video
    for frame_file in frame_files:
        frame_path = os.path.join(frames_folder, frame_file)
        frame = cv2.imread(frame_path)
        out.write(frame)

    # Release the video writer
    out.release()
    print(f"Video saved as {output_video}")




# Load Models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

prep_net = PrepNetwork()
hide_net = HidingNetwork()
reveal_net = RevealNetwork()

prep_net.load_state_dict(remove_module_prefix(torch.load("models/model_9.2.4/prep_net.pth", map_location=device)))
hide_net.load_state_dict(remove_module_prefix(torch.load("models/model_9.2.4/hide_net.pth", map_location=device)))
reveal_net.load_state_dict(remove_module_prefix(torch.load("models/model_9.2.4/reveal_net.pth", map_location=device)))

prep_net.to(device)
hide_net.to(device)
reveal_net.to(device)



# Video to Frames
video_file_cover = "video_processing/input_video1.mp4"  
extract_frames(video_file_cover, output_folder="video_processing/frames_cover") 
video_file_secret = "video_processing/input_video2.mp4"  
extract_frames(video_file_secret, output_folder="video_processing/frames_secret")
# Define transforms
transform1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)) 
])
transform2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128)) 
])
# Create dataset and dataloader"video_processing/frames_cover"
dataset = VideoFramesDataset("video_processing/frames_cover", "video_processing/frames_secret", transform1, transform2)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False, drop_last=True)  # No shuffling to maintain order
print("Dataset and DataLoader created successfully.")

# Inference
process_video(prep_net, hide_net, reveal_net, dataloader, loss_function, device=device)
# Reconstruct Video
create_video_from_frames(frames_folder="video_processing/stego_frames")

