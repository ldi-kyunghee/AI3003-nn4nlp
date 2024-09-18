import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import time
import random
import numpy as np
from tqdm import tqdm

# Functions to read in the corpus
w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
UNK = w2i["<unk>"]

def read_dataset(filename):
    with open(filename, "r") as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            yield ([w2i[x] for x in words.split(" ")], t2i[tag])

# Read in the data
train_data = list(read_dataset("../data/classes/train.txt"))
w2i = defaultdict(lambda: UNK, w2i)
dev_data = list(read_dataset("../data/classes/test.txt"))
nwords = len(w2i)
ntags = len(t2i)

# Define the model
EMB_SIZE = 64
HID_SIZE = 64

class RNNModel(nn.Module):
    def __init__(self, nwords, ntags, emb_size, hidden_size):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(nwords, emb_size)
        self.rnn = nn.RNN(emb_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, ntags)
    
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        # Apply Xavier initialization to all linear and recurrent layers
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)  # Initialize biases to zero
    
    def forward(self, sentence):
        embeds = self.embedding(sentence) # [len(sentence) x emb_size]
        rnn_out, _ = self.rnn(embeds)# [len(sentence) x hidden_size]
        logits = self.fc(rnn_out[-1])
        return logits

model = RNNModel(nwords, ntags, EMB_SIZE, HID_SIZE)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
for ITER in range(100):
    # Perform training
    random.shuffle(train_data)
    train_loss = 0.0
    start = time.time()
    model.train()

    for words, tag in tqdm(train_data):
        words = torch.tensor(words)
        tag = torch.tensor(tag)

        logits = model(words)
        loss = criterion(logits, tag)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"iter {ITER}: train loss/sent={train_loss / len(train):.4f}, time={time.time() - start:.2f}s")

    # Perform evaluation
    model.eval()
    test_correct = 0.0
    with torch.no_grad():
        for words, tag in tqdm(dev_data):
            words = torch.tensor(words)
            logits = model(words)
            predict = torch.argmax(logits).item()
            if predict == tag:
                test_correct += 1
    print(f"iter {ITER}: test acc={test_correct / len(dev):.4f}")