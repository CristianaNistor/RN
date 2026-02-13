import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
import pandas as pd
import pickle, os
from torch import Tensor

BATCH_SIZE = 64
INPUT_SIZE = 784
HIDDEN_SIZE_1 = 512
HIDDEN_SIZE_2 = 128
OUTPUT_SIZE = 10
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
DROPOUT_PROB = 0.05

class ExtendedMNISTDataset(Dataset):
    def __init__(self, root="/kaggle/input/fii-nn-2025-homework-4", train=True):
        file = "extended_mnist_train.pkl" if train else "extended_mnist_test.pkl"
        file = os.path.join(root, file)
        with open(file, "rb") as fp:
            self.data = pickle.load(fp)

        images, labels = zip(*self.data)
        # float32 e standard pt GPU
        images = np.array(images, dtype=np.float32) / 255.0 # convertesc imaginile intr-un array NumPy si le normalizez
        images = torch.from_numpy(images) # transform in tensor PyTorch de forma (50 000, 28, 28)
        labels = torch.tensor(labels, dtype=torch.long) # labels devin torch.long (necesar pentru CrossEntropyLoss)
        # Tensor = colecție de numere organizate într-un număr arbitrar de dimensiuni, deci un fel de array NumPy

        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.images[i].view(-1), self.labels[i] # image.view(-1) transforma imaginea 2D in vector de 784 elemente

train_loader = DataLoader(ExtendedMNISTDataset(train=True), batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(ExtendedMNISTDataset(train=False), batch_size=BATCH_SIZE, shuffle=False)

class MLP(nn.Module): # nn.Module e clasa de baza pt modele
    def __init__(self, input_size=INPUT_SIZE, hidden1=HIDDEN_SIZE_1, hidden2=HIDDEN_SIZE_2, output_size=OUTPUT_SIZE, dropout_prob=DROPOUT_PROB):
        super().__init__()

        # straturile nn.Linear folosesc o inițializare foarte bună implicit (Kaiming uniform pentru ReLU)
        self.fc1 = nn.Linear(input_size, hidden1) # strat liniar: 784 -> 512
        # w1: 784x512, b1: 1x512
        self.fc2 = nn.Linear(hidden1, hidden2) # strat liniar: 512 -> 128
        # w2: 512x128, b2: 1x128
        self.out = nn.Linear(hidden2, output_size) # strat liniar: 128 -> 10
        # w3: 128x10, b3: 1x10
        self.dropout = nn.Dropout(dropout_prob) # aplic dropout: dezactivez aleator neuroni

    def forward(self, x: Tensor) -> Tensor: # def cum trece inputul prin retea
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.out(x)
        return x
    
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

device = get_device()

model = MLP().to(device) # mut modelul pe device, la fel si tensorii X, y

criterion = nn.CrossEntropyLoss() # claculez loss-ul
# Optimizers apply the gradients calculated by the Autograd engine to the weights, using their own optimization technique
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001) # trateaza regularizarea L2 mai eficient decat Adam
# ajustează rata de învățare (LR) pentru fiecare parametru individual (greutate sau bias)
# Separă Regularizarea L2 de optimizatorul în sine și o aplică direct la greutăți (după ce gradientul a fost calculat)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) # reduce lr la fiecare step_size epoci; la fiecare 10 epoci lr se injumatateste

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad() # curata gradientii anteriori
        outputs = model(X) # forward pass
        loss = criterion(outputs, y) # loss-ul e un scalar
        loss.backward() # calculeaza gradientii pt fiecare parametru
        optimizer.step()  # updateaza parametrii folosind gradientii calculati si AdamW
        total_loss += loss.item()
        
    return total_loss / len(loader) # media per batch

for epoch in range(NUM_EPOCHS):
    loss = train(model, train_loader, optimizer, criterion, device)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {loss:.4f}")
    scheduler.step()

model.eval()
preds = []
with torch.no_grad(): # dezactivez mecanismul de calcul al gradientului, deoarece nu mai e nevoie de backward pass
    for X, _ in test_loader:
        X = X.to(device)
        outputs = model(X) # obtin predictiile modelului
        _, predicted = torch.max(outputs, 1) # dim=1 repr max pe coloanele unei linii
        preds.extend(predicted.cpu().numpy())

df = pd.DataFrame({"ID": range(len(preds)), "target": preds})
df.to_csv("submission.csv", index=False)
df.head()