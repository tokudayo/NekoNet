import torch
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


import yaml
def load_yaml(path):
    file = open(path, 'r')
    options = yaml.load(file, Loader=yaml.FullLoader)
    file.close()
    return options

# Horrible and unsafe way to freeze/unfreeze layers, but I'm desperate

def freeze(model, layers):
    if layers == 'all':
        for param in model.parameters(): param.requires_grad = False
    else: 
        for layer in layers:
            exec(f'''for param in model.{layer}.parameters():\n    param.requires_grad = False''')

def unfreeze(model, layers):
    if layers == 'all':
        for param in model.parameters(): param.requires_grad = True
    else: 
        for layer in layers:
            exec(f'''for param in model.{layer}.parameters():\n    param.requires_grad = True''')