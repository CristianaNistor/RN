import pickle
import os
import pandas as pd
import numpy as np

# load data
train_file = "extended_mnist_train.pkl"
test_file = "extended_mnist_test.pkl"

with open(train_file, "rb") as fp:
    train = pickle.load(fp)

with open(test_file, "rb") as fp:
    test = pickle.load(fp)

# flatten images
train_data = []
train_labels = []
for image, label in train:
    train_data.append(image.flatten())
    train_labels.append(label)

test_data = []
for image, label in test:
    test_data.append(image.flatten())

# normalizarea pixelilor (0–1)
train_data = np.array(train_data, dtype=np.float32) / 255.0
train_labels = np.array(train_labels, dtype=np.int32)

test_data = np.array(test_data, dtype=np.float32) / 255.0

# transformarea vectorului de label-uri im matrice
y_train_encoded = np.eye(10)[train_labels]

# params
input_dim = 784 # 28 x 28 pixeli
output_dim = 10 # 10 clase de cifre
learning_rate = 0.1
epochs = 100 # iteratii
batch_size = 64 # 64 e recomandat in curs

np.random.seed(40)
W = np.random.randn(input_dim, output_dim) * 0.01 # inmultesc cu 0.1 pt ca altfel valorile ar fi f mari
b = np.zeros((1, output_dim)) # il initializez cu 0


def softmax(z):
    # transforma scorurile brute z in probabilitati
    # scad din exponent maximul pt a nu face overflow numeric din cauza exponentialei
    e = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

def cross_entropy(y, p):
    # calc eroarea (loss-ul) dintre valorile reale (y) si cele prezise (p)
    return -np.mean(np.sum(y * np.log(p), axis=1))

def accuracy(y, p):
    # proportia de exemple clasificate corect
    return np.mean(np.argmax(y, axis=1) == np.argmax(p, axis=1))

# batch training
n = train_data.shape[0]
for ep in range(epochs):
    # shuffle
    perm = np.random.permutation(n)
    X = train_data[perm] # reordonez datele
    Y = y_train_encoded[perm] # reordonez label-urile

    for i in range(0, n, batch_size):
        Xb = X[i:i+batch_size] # extrag un batch de date
        Yb = Y[i:i+batch_size] # extrag batch-ul corespunzator de label-uri

        z = Xb.dot(W) + b
        p = softmax(z)

        dz = (p - Yb) / Xb.shape[0] # gradientul pierderii fata de z
        dW = Xb.T.dot(dz) # gradientul pt fiecare greutate din W
        db = dz.sum(axis=0, keepdims=True) # gradientul pt bias

        # actualizez 
        W = W -  learning_rate * dW
        b = b - learning_rate * db

    # eval
    p_all = softmax(train_data.dot(W) + b)
    print(f"Epoch {ep+1}/{epochs} — loss {cross_entropy(y_train_encoded, p_all):.4f} acc {accuracy(y_train_encoded, p_all)*100:.2f}%")

# predict and save
preds = np.argmax(softmax(test_data.dot(W) + b), axis=1)
pd.DataFrame({"ID": np.arange(len(preds)), "target": preds}).to_csv("submission.csv", index=False)
print("submission.csv saved")