import torch
from model import Net
from utils import DataPipeline
import torchvision.transforms as T
from triplet_loss import TripletLoss

device = 'cuda' if torch.cuda.is_available() else 'cpu'
epoch = 20
batch_size = 32

transform = T.Compose([T.Resize((224, 224)),
                       lambda x : x/255.0,
                       T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
loader = DataPipeline('./data', batch_size, tsnf = transform)

model = Net()
model = model.to(device)
criterion = TripletLoss(device)
optimizer = torch.optim.Adam(model.parameters())

for ep in range(epoch):
    print(f'Epoch {ep + 1}')
    for x, y in loader.generator():
        x = x.to(device)
        y = y.to(device)
        print("FF")
        # FF
        embeddings = model(x)
        loss = criterion(embeddings, y)
        print(f"loss = {loss.item()}")
        print("BP")
        # BP
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
