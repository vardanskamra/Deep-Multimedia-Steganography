import os
import torch
import torchvision
import torchvision.transforms as transforms
from collections import OrderedDict
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, url_for
from models import PrepNetwork, HidingNetwork, RevealNetwork

def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = value
    return new_state_dict

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dir_path = os.path.dirname(os.path.realpath(__file__))
prep_net = PrepNetwork()
hide_net = HidingNetwork()
reveal_net = RevealNetwork()
prep_net.load_state_dict(remove_module_prefix(torch.load(os.path.join(dir_path, "prep_net.pth"), map_location=device)))
hide_net.load_state_dict(remove_module_prefix(torch.load(os.path.join(dir_path, "hide_net.pth"), map_location=device)))
reveal_net.load_state_dict(remove_module_prefix(torch.load(os.path.join(dir_path, "reveal_net.pth"), map_location=device)))

inference_transform_1 = transforms.Compose([
    transforms.Resize((128, 128))])
inference_transform_2 = transforms.Compose([
    transforms.Resize((256, 256))])

def inference(cover_path: str,
              secret_path: str,
              prep_net: torch.nn.Module,
              hide_net: torch.nn.Module,
              reveal_net: torch.nn.Module,
              device: str = None,
              output_path: str = None):
    """
    If secret_path is not None, hides secret in cover and writes stego to output_path.
    If secret_path is None, treats cover_path as a stego image and writes revealed secret.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # load images
    cover = torchvision.io.read_image(cover_path).float()/255
    cover = inference_transform_2(cover)
    cover = cover.unsqueeze(0).to(device)
    is_hide = secret_path is not None
    if is_hide:
        secret = torchvision.io.read_image(secret_path).float()/255
        secret = inference_transform_1(secret)
        secret = secret.unsqueeze(0).to(device)
    prep_net.to(device).eval()
    hide_net.to(device).eval()
    reveal_net.to(device).eval()
    with torch.inference_mode():
        if is_hide:
            # hide case
            secret_prepared = prep_net(secret)
            stego = hide_net(cover, secret_prepared)
            output = stego
        else:
            # reveal case: cover_path is actually stego
            stego = cover
            secret_revealed = reveal_net(stego)
            output = secret_revealed
        # clamp to [0,1]
        output = output.clamp(0.0, 1.0)
        # save
        if output_path is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # torchvision.utils will tile a batch, but here batch_size=1
            torchvision.utils.save_image(output, output_path)
        return output
        
# Configuration
dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = os.path.join(dir_path, 'static', 'results')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/hide', methods=['GET', 'POST'])
def hide():
    if request.method == 'POST':
        cover = request.files.get('cover')
        secret = request.files.get('secret')
        if cover and secret and allowed_file(cover.filename) and allowed_file(secret.filename):
            cover_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(cover.filename))
            secret_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(secret.filename))
            cover.save(cover_path)
            secret.save(secret_path)
            # Run inference to hide secret
            stego_path = os.path.join(app.config['UPLOAD_FOLDER'], 'stego.png')
            inference(cover_path, secret_path, prep_net, hide_net, reveal_net, device, output_path=stego_path)
            return render_template('hide.html', stego_url=url_for('static', filename=f'results/stego.png'))
    return render_template('hide.html', stego_url=None)

@app.route('/reveal', methods=['GET', 'POST'])
def reveal():
    if request.method == 'POST':
        stego = request.files.get('stego')
        if stego and allowed_file(stego.filename):
            stego_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(stego.filename))
            stego.save(stego_path)
            # Run inference to reveal secret
            revealed_path = os.path.join(app.config['UPLOAD_FOLDER'], 'revealed.png')
            inference(stego_path, None, prep_net, hide_net, reveal_net, device, output_path=revealed_path)
            return render_template('reveal.html', revealed_url=url_for('static', filename=f'results/revealed.png'))
    return render_template('reveal.html', revealed_url=None)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)