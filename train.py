import argparse
import torch, os
from matplotlib import pyplot as plt
from model import *
from utils import *

from dataloader import DataLoader
from loss import TripletLoss, TripletLossWithGOR

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, help='Experiment output path', required=True)
    parser.add_argument('-w', '--weight', type=str, help='Load initial model weight', required=False)
    parser.add_argument('-c', '--config', type=str, default='./output', help='Target directory', required=True)
    args = parser.parse_args()
    return args

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train():
    # Configuration
    epochs = 20
    batch_size = 32
    val_step = None
    train_path = './data/cropped224'
    val_path = None
    test_path = None
    out_dir = './exp/run1'

    # Triplet loss with GOR conf
    alpha_gor = 1.0
    margin = 1.0

    model = MobileNetV3L64()
    model = model.to(device)
    criterion = TripletLoss(device)
    optimizer = torch.optim.Adam(model.parameters())

    # ImageNet preprocessing of [0; 1] (3, H, W) tensor input
    transform = T.Compose([#T.Resize((224, 224)),
                        lambda x : x/255.0,
                        T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
    loader = DataLoader(train_path, batch_size, tsnf = transform)

    # Training
    ## Continue/start new training
    try:
        os.mkdir(out_dir)
    except:
        pass
    if os.path.isfile(out_dir + '/last.pt'):
        state = torch.load(out_dir + '/last.pt')
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        current = state['epoch'] + 1
        print(f'Resume training at epoch {current + 1}')
    else:
        current = 0

    loss_epoch = []
    ## Training loop
    for ep in range(current, epochs):
        # Processing per epoch
        loss_batch = []
        print(f'Epoch {ep + 1}')
        for index, (x, y) in enumerate(loader.generator()):
            # Processing per batch
            x = x.to(device)
            y = y.to(device)
            embeddings = model(x)
            loss = criterion(embeddings, y)
            loss_batch.append(loss.item())
            print(f'Batch {index + 1} loss = {loss.item()}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_epoch.append(np.average(loss_batch))
        print(f'Epoch {ep} loss = {loss_epoch[-1]}')
        save_model(model, optimizer, ep, out_dir + f'/last.pt')

    torch.save(model, out_dir + '/final.pt')
    # Result
    plt.plot(loss_epoch)
    plt.show()

if __name__=="__main__":
    args = parse_args()
    train(args)