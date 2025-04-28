import torch
import torchvision
import torchvision.transforms as transforms
from utils.inference import remove_module_prefix
from utils.visualizations import visualize_images, differentiate_images
from models.model_9 import PrepNetwork, HidingNetwork, RevealNetwork

device=torch.device("cpu")

prep_net = PrepNetwork()
hide_net = HidingNetwork()
reveal_net = RevealNetwork()

prep_net.load_state_dict(remove_module_prefix(torch.load("models/model_9.2.4/prep_net.pth", map_location=device)))
hide_net.load_state_dict(remove_module_prefix(torch.load("models/model_9.2.4/hide_net.pth", map_location=device)))
reveal_net.load_state_dict(remove_module_prefix(torch.load("models/model_9.2.4/reveal_net.pth", map_location=device)))

prep_net.to(device)
hide_net.to(device)
reveal_net.to(device)

inference_transform_1 = transforms.Compose([
    transforms.Resize((128, 128))
])

inference_transform_2 = transforms.Compose([
    transforms.Resize((256, 256))
])

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
    cover, secret = inference_transform_2(cover), inference_transform_1(secret)
    
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
        
        differentiate_images(cover.cpu().squeeze(0), stego.cpu().squeeze(0))

inference(cover_path="images/burger.jpeg", 
           secret_path="images/fries.jpg", 
           prep_net=prep_net, 
           hide_net=hide_net, 
           reveal_net=reveal_net,
           device = "cpu")
 
