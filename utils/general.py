import torch, yaml
import numpy as np
from matplotlib import pyplot as plt

def params_info(net):
    total = 0
    trainable = 0
    for param in net.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    
    print(f'Total parameters: {total}')
    print(f'Trainable: {trainable}')
    print(f'Non-trainable: {total - trainable}')


def batch_to_images(X, y):
    X = X.to('cpu').numpy()
    batch_size = X.shape[0]
    r = c = int(np.ceil(np.sqrt(batch_size)))
    fig = plt.figure(figsize=(r, c))
    for index, img in enumerate(X):
        # Adds a subplot at the 1st position
        fig.add_subplot(r, c, index + 1)
        # showing image
        plt.imshow(img.transpose(1, 2, 0))
        plt.axis('off')
        plt.title(y[index].item())
    plt.show()


def save_model(model, opt, ep, path):
    state = {
        'epoch': ep,
        'state_dict': model.state_dict(),
        'optimizer': opt.state_dict(),
    }
    torch.save(state, path)


def load_yaml(path):
    file = open(path, 'r')
    options = yaml.load(file, Loader=yaml.FullLoader)
    file.close()
    return options


# Horrible and unsafe way to freeze/unfreeze layers, but I'm desperate
def freeze(model, layers):
    if layers is None: return
    if layers == 'all':
        print("Freezing all params...")
        for param in model.parameters(): param.requires_grad = False
    else:
        if type(layers) == str: layers = [layers]
        for layer in layers:
            print(f"Freezing {layer}")
            exec(f'''for param in model.{layer}.parameters():\n    param.requires_grad = False''')


def unfreeze(model, layers):
    if layers is None: return
    if layers == 'all':
        print("Unfreezing all params...")
        for param in model.parameters(): param.requires_grad = True
    else:
        if type(layers) == str: layers = [layers]
        for layer in layers:
            print(f"Unfreezing {layer}")
            exec(f'''for param in model.{layer}.parameters():\n    param.requires_grad = True''')


def select_model(choice):
    models = getattr( __import__('models'), 'descriptors')
    try:
        class_ = getattr(models, choice)
        return class_()
    except:
        print(f'Class {choice} not found. You can define a your model in models.py')


def onnx_export(model, name):
    import torch.onnx
    import os
    os.makedirs('./export')
    fpath = os.path.join('./export', name)
    if os.path.isfile(fpath):
        print("Warning: file already exists. Continue? (y/n)")
        t = input()
        if t[0].lower() != 'y':
            return
        
    batch_size = 1
    channel = 3
    height = width = 224
    dummy_input = torch.randn(batch_size, channel, height, width)

    torch.onnx.export(model,                     # model being run
                    dummy_input,               # model input (or a tuple for multiple inputs)
                    fpath,                # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=12,          # the ONNX version to export the model to (default 9). 12 de cung version voi YOLOv5
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})


# For backward compatibility
def load_weight(model, weightpath):
    ref = torch.load(weightpath)
    model.load_state_dict(ref)

import urllib, os
def attempt_download(localpath, dllink):
    if not os.path.isfile(localpath):
        print(f'{localpath} not found, downloading from {dllink}')
        urllib.request.urlretrieve(dllink, localpath)
