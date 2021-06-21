import torch, os
from matplotlib import pyplot as plt
from model import *
from utils import *

from dataloader import DataLoader
from loss import TripletLossWithGOR

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Configuration
from cfg import *
model = MobileNetV3L64().to(device)
criterion = TripletLossWithGOR(device, alpha_gor = alpha_gor, margin = margin)
optimizer = torch.optim.Adam(model.parameters())

loader = DataLoader(train_path, batch_size, tsnf = model.transform)

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
