import torch
from utils.inference import inference, remove_module_prefix
from models.model_4 import PrepNetwork, HidingNetwork, RevealNetwork

device=torch.device("cpu")

prep_net = PrepNetwork()
hide_net = HidingNetwork()
reveal_net = RevealNetwork()

prep_net.load_state_dict(remove_module_prefix(torch.load("models/model_4/prep_net.pth", map_location=device)))
hide_net.load_state_dict(remove_module_prefix(torch.load("models/model_4/hide_net.pth", map_location=device)))
reveal_net.load_state_dict(remove_module_prefix(torch.load("models/model_4/reveal_net.pth", map_location=device)))

prep_net.to(device)
hide_net.to(device)
reveal_net.to(device)

inference(cover_path="images/burger.jpeg", 
          secret_path="images/fries.jpg", 
          prep_net=prep_net, 
          hide_net=hide_net, 
          reveal_net=reveal_net,
          device = "cpu")
