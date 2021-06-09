import torch
from model import Net
from utils import DataPipeline
import torchvision.transforms as T
from triplet_loss import TripletLoss

dl = DataPipeline('./data', 16, tsnf = T.Compose([T.Resize((224, 224)),
                                                  lambda x : x/255.0,
                                                  T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])]))

model = Net()
criterion = TripletLoss('cpu')
optimizer = torch.optim.Adam(model.parameters())

for i in range(20):
    print(f"Batch {i + 1}")
    x, y = dl.get()
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
