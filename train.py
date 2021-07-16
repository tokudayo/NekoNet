import argparse
import torch, os
from matplotlib import pyplot as plt
from tqdm import tqdm
from models import *
from utils import *
from loss import *
from dataloader import DataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.yaml', help='Training configuration', required=False)
    args = parser.parse_args()
    return args

def train(cfg_path):
    # Load configuration file
    try:
        opt = load_yaml(cfg_path)
    except:
        print(f"Failed to load configuration file {cfg_path}.")
        return

    # General conf.
    epochs = opt['epochs']
    batch_size = opt['batch_size']
    train_path = opt['train_data']
    val_path = opt['val_data']
    out_dir = opt['out_dir']

    # Model conf.
    if opt['weight'] is not None:
        try:
            model = torch.load(opt['weight'])
        except:
            print(f"Failed to load weight from {opt['weight']}.")
            return
    else:
        model = select_model(opt['model'])
        if model is None: return
    model = model.to(device)
    print(model)
    freeze(model, opt['freeze'])
    unfreeze(model, opt['unfreeze'])
    params_info(model)
    
    # Triplet loss with GOR conf.
    alpha_gor = 1.0
    margin = 1.0
    if 'alpha_gor' in opt.keys(): alpha_gor = opt['alpha_gor']
    if 'loss_margin' in opt.keys(): margin = opt['loss_margin']
    if opt['loss_type'] == 'semihard':
        criterion = SemiHardTripletLossWithGOR(margin, alpha_gor=alpha_gor)
    elif opt['loss_type'] == 'hard':
        criterion = HardTripletLossWithGOR(margin, alpha_gor=alpha_gor)
    elif opt['loss_type'] == 'hardest':
        criterion = HardTripletLossWithGOR(margin, alpha_gor=alpha_gor, hardest=True)
    else:
        print("Unknown loss metric.")
        return
    print(f"Loss metric: {opt['loss_type']} triplet loss.")
    optimizer = torch.optim.Adam(model.parameters())
    

    # ImageNet preprocessing of [0; 255] (3, H, W) RGB tensor input
    transform = T.Compose([T.Resize((224, 224)),
                        lambda x : x/255.0,
                        T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
    loader = DataLoader(train_path, tsnf = transform, cache_path='samplecache.pt')

    # Training
    ## Continue/start new training
    try:
        os.makedirs(out_dir)
    except:
        pass
    if os.path.isfile(out_dir + '/last.pt'):
        state = torch.load(out_dir + '/last.pt')
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        current = state['epoch'] + 1
        loss_epoch = torch.load(out_dir + '/loss.pt')
        print(f'Resuming training at epoch {current + 1}')
    else:
        current = 0
        loss_epoch = []
    
    ## Training loop
    for ep in tqdm(range(current, epochs), desc='Overall progrress'):
        # Processing per epoch
        loss_batch = []
        #print(f'Epoch {ep + 1}')
        for index, (x, y) in tqdm(enumerate(loader.generator(batch_size)), desc=f'Epoch {ep + 1}', leave=False, total=int(loader.count_all()/batch_size)):
            # Processing per batch
            x = x.to(device)
            y = y.to(device)
            embeddings = model(x)
            loss = criterion(embeddings, y)
            loss_batch.append(loss.item())
            # print(f'Batch {index + 1} loss = {loss.item()}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_epoch.append(np.average(loss_batch))
        print(f'Epoch {ep + 1} loss = {loss_epoch[-1]}')
        save_model(model, optimizer, ep, out_dir + f'/last.pt')
        torch.save(loss_epoch, out_dir + '/loss.pt')

    torch.save(model, out_dir + '/final.pt')

    # Result
    plt.plot(loss_epoch)
    plt.savefig(os.path.join(out_dir,'loss.png'))

if __name__=="__main__":
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {device}")
    train(args.config)