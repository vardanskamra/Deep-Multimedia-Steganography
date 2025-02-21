import torch
import torchvision
from utils.transforms import inference_transform
from utils.visualizations import visualize_images

def inference(cover_path, 
              secret_path, 
              prep_net: torch.nn.Module, 
              hide_net: torch.nn.Module, 
              reveal_net: torch.nn.Module,
              device = None):
    
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    cover = torchvision.io.read_image(str(cover_path)).type(torch.float32)
    secret = torchvision.io.read_image(str(secret_path)).type(torch.float32)
    
    cover, secret = cover/255, secret/255
    cover, secret = inference_transform(cover), inference_transform(secret)
    
    cover, secret = cover.unsqueeze(0).to(device), secret.unsqueeze(0).to(device)
    
    prep_net.eval()
    hide_net.eval()
    reveal_net.eval()
    
    with torch.inference_mode():
        
        secret_prepared = prep_net(secret)
        stego = hide_net(cover, secret_prepared)
        secret_revealed = reveal_net(stego)
        
        visualize_images(cover.cpu().squeeze(0), 
                        secret.cpu().squeeze(0), 
                        stego.cpu().squeeze(0), 
                        secret_revealed.cpu().squeeze(0))

    