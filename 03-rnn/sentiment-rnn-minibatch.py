import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import time
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

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
w2i = defaultdict(lambda: UNK, w2i)  # Now UNK is used for unknown words
dev_data = list(read_dataset("../data/classes/test.txt"))
nwords = len(w2i)
ntags = len(t2i)

# Custom dataset class to work with DataLoader
class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# Collate function for padding sentences in each batch
def collate_fn(batch):
    sentences, tags = zip(*batch)
    sentences = [torch.tensor(sent) for sent in sentences]
    tags = torch.tensor(tags)
    padded_sentences = pad_sequence(sentences, batch_first=True, padding_value=UNK)
    return padded_sentences, tags

# Define DataLoader
BATCH_SIZE = 32
train_loader = DataLoader(TextDataset(train_data), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(TextDataset(dev_data), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

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
    
    def forward(self, sentences):
        embeds = self.embedding(sentences)  # [batch_size x len(sentences[1]) x emb_size]
        rnn_out, _ = self.rnn(embeds)       # [batch_size x len(sentences[1]) x hidden_size]
        logits = self.fc(rnn_out[:, -1, :]) # Use the last hidden state for classification
        return logits

model = RNNModel(nwords, ntags, EMB_SIZE, HID_SIZE)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
max_test_accuracy = 0.0
for ITER in range(100):
    # Perform training
    train_loss = 0.0
    start = time.time()
    model.train()

    for sentences, tags in tqdm(train_loader):
        logits = model(words)
        loss = criterion(logits, tags)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"iter {ITER}: train loss/sent={train_loss / len(train_loader):.4f}, time={time.time() - start:.2f}s")

    # Perform evaluation
    model.eval()
    test_correct = 0.0
    with torch.no_grad():
        for sentences, tags in tqdm(dev_loader):
            logits = model(sentences)
            predict = torch.argmax(logits, dim=1)
            test_correct += (predict == tags).sum().item()

    test_accuracy = test_correct / len(dev_data)
    if max_test_accuracy < test_accuracy:
        max_test_accuracy = test_accuracy
    print(f"iter {ITER}: test acc={test_accuracy:.4f}")
    
print("max test acc=%.4f" % (max_test_accuracy))