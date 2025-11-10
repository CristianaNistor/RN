import numpy as np
import pickle
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

# incarcarea datelor
train_file = "extended_mnist_train.pkl"
test_file = "extended_mnist_test.pkl"

with open(train_file, "rb") as fp:
    train = pickle.load(fp)
with open(test_file, "rb") as fp:
    test = pickle.load(fp)

train_data, train_labels = [], []
for image, label in train:
    train_data.append(image.flatten())
    train_labels.append(label)

test_data = [image.flatten() for image, _ in test]

# normalizarea pixelilor
X = np.array(train_data, dtype=np.float32) / 255.0 # 10.000 x 784
y = np.array(train_labels, dtype=np.int64)
X_test = np.array(test_data, dtype=np.float32) / 255.0

# impartirea train/validation 
# 10% din datele de care dispun pentru antrenare le voi aloca validarii
# random state e pentru a obtine aceeasi impartire a datelor la fiecare rulare
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=40)

# parametri
input_size = 784
hidden_size = 100
output_size = 10

# initializarea weight-urilor cu Xavier normal
rng = np.random.default_rng(40) # creeaza un generator de numere aleatoare
def xavier_init(fan_in, fan_out):
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return rng.normal(0, std, (fan_in, fan_out))

W1 = xavier_init(input_size, hidden_size) # 784 x 100
b1 = np.zeros((1, hidden_size)) # 1 x 100
W2 = xavier_init(hidden_size, output_size) # 100 x 10
b2 = np.zeros((1, output_size)) # 1 x 10

# hiperparametri
epochs = 50
batch_size = 32
lr = 0.1
dropout_rate = 0.2

# functii
def relu(x): return np.maximum(0, x)
def relu_deriv(x): return (x > 0).astype(np.float32)
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)
def accuracy(preds, labels): return np.mean(preds == labels)

# antrenare
for epoch in range(epochs):
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    X_train, y_train = X_train[indices], y_train[indices]

    total_loss, total_acc, num_batches = 0, 0, 0

    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i+batch_size] # 32 x 784
        y_batch = y_train[i:i+batch_size]

        # forward
        z1 = X_batch.dot(W1) + b1 # 32 x 100
        a1 = relu(z1)

        # dropout doar in training
        dropout_mask = (np.random.rand(*a1.shape) > dropout_rate).astype(np.float32)
        # *a1.shape despacheteaza tupla (batch_size, hidden_dim) in argumente separate pentru np.random.rand, adica np.random.rand(batch_size, hidden_dim)
        # generez o matrice de 32 x 100 de valori random intre 0 si 1 si pastrez activi doar neuronii care corespund unei valori > 0.2 
        # => o matrice booleeana => matrice cu float-uri 0.0 sau 1.0
        
        a1 = a1 * dropout_mask / (1 - dropout_rate)
        # aplic dropout si scalez activarile pt a pastra magnitudinea asteptata
        # astfel media activa ramane aprox aceeasi ca in timpul testului, cand toti neuronii sunt activi 

        z2 = a1.dot(W2) + b2 # 32 x 10
        logits = torch.tensor(z2, dtype=torch.float32)
        labels_t = torch.tensor(y_batch, dtype=torch.long)

        loss = F.cross_entropy(logits, labels_t)
        total_loss += loss.item()

        # backprop
        probs = softmax(z2) # 32 x 10
        # softmax tansforma scorurile in probabilitati pentru fiecare clasa

        # np.arange(len(y_batch)) creeaza o secventa de indici pentru fiecare rand din batch (0, 1, ..., 31)
        # probs[np.arange(len(y_batch)), y_batch] acceseaza exact elementele din probs corespunzatoare claselor corecte pentru fiecare exemplu din batch
        # o lista de tuple de forma (i, label_corect)
        probs[np.arange(len(y_batch)), y_batch] -= 1
        # din softmax de z2 vreau sa scad 1 de pe fiecare rand, dar de pe pozitia cu label-ul corect
        # adica softmax(z2) -y one-hot pt ca atat e derivata cross entropiei
        probs /= len(y_batch)
        # probs = dL/dz2 = 1/batch_size (softmax(z2)-1)

        dW2 = a1.T.dot(probs) # 100 x 10
        # dW2 = dL/dw2 = a1.T dL/dz2 = a1.T probs
        db2 = np.sum(probs, axis=0, keepdims=True)
        # db2 = dL/db2 = suma din dL/dz2(i) = suma valorilor de pe coloane si astfel obtin dimensiunea 1 x nr de coloane din probs = 1 x 10

        da1 = probs.dot(W2.T) # da1 = dL/da1 = w2.T x dL/dz2 = w2.T x probs
        dz1 = da1 * relu_deriv(z1) #dz1 = dL/dz1 = dL/da1 x da1/dz1 = da1 x relu_derivat de z1
        dz1 *= dropout_mask / (1 - dropout_rate) # dz1 = dz1 x dropout_mask / (1- dropout_mask) 
        # in forward am inmultit a1 cu dropout_mask / (1 - dropout_rate) ca sa mentinem magnitudinea medie; in backprop facem la fel

        dW1 = X_batch.T.dot(dz1) # dL/dw1 = x.T x dL/dz1 = x.T x dz1
        db1 = np.sum(dz1, axis=0, keepdims=True) 
        # db1 = dL/db1 = suma din dL/dz1(i) = suma valorilor de pe coloane din dz1 => 1 x nr de col din dz1 = 1 x 100

        # gradient descent simplu
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

        preds = np.argmax(z2, axis=1)
        total_acc += accuracy(preds, y_batch)
        num_batches += 1

    # validare
    z1_val = X_val.dot(W1) + b1
    a1_val = relu(z1_val)
    z2_val = a1_val.dot(W2) + b2
    logits_val = torch.tensor(z2_val, dtype=torch.float32)
    labels_val = torch.tensor(y_val, dtype=torch.long)
    val_loss = F.cross_entropy(logits_val, labels_val).item()
    val_acc = accuracy(np.argmax(z2_val, axis=1), y_val)

    print(f"Epoch {epoch+1:02d}/{epochs}: "
          f"Train loss={total_loss/num_batches:.4f}, acc={total_acc/num_batches:.4f}, "
          f"Val loss={val_loss:.4f}, acc={val_acc:.4f}")

# testare
z1_test = X_test.dot(W1) + b1
a1_test = relu(z1_test)
z2_test = a1_test.dot(W2) + b2
preds = np.argmax(z2_test, axis=1)

pd.DataFrame({"ID": np.arange(len(preds)), "target": preds}).to_csv("submission.csv", index=False)
print("\n File 'submission.csv' saved.")