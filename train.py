import torch
from torchvision.transforms import transforms
from model import Net
from utils import *
import torchvision.transforms as T
from triplet_loss import TripletLoss
from matplotlib import pyplot as plt

# Model and training configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epoch = 20
batch_size = 16
val_step = 3
out_dir = './exp/sample'

transform = T.Compose([lambda x : x/255.0,
                       T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
loader = DataPipeline('./data', batch_size, tsnf = transform, image_size=(224, 224))

model = Net()
model = model.to(device)
criterion = TripletLoss(device)
optimizer = torch.optim.Adam(model.parameters())

# Training
loss_epoch = []
for ep in range(epoch):
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
    print(f'Epoch {ep + 1} loss = {loss_epoch[-1]}')

# Result
plt.plot(loss_epoch)
plt.show()
