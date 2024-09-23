import time
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu

# Data paths
train_src_file = "../data/parallel/train.ja"
train_trg_file = "../data/parallel/train.en"
dev_src_file = "../data/parallel/dev.ja"
dev_trg_file = "../data/parallel/dev.en"
test_src_file = "../data/parallel/test.ja"
test_trg_file = "../data/parallel/test.en"

w2i_src = defaultdict(lambda: len(w2i_src))
w2i_trg = defaultdict(lambda: len(w2i_trg))

def read(fname_src, fname_trg):
    with open(fname_src, "r") as f_src, open(fname_trg, "r") as f_trg:
        for line_src, line_trg in zip(f_src, f_trg):
            sent_src = [w2i_src[x] for x in line_src.strip().split() + ['</s>']]
            sent_trg = [w2i_trg[x] for x in ['<s>'] + line_trg.strip().split() + ['</s>']]
            yield (sent_src, sent_trg)

# Build vocab and load data
train_data = list(read(train_src_file, train_trg_file))
dev_data = list(read(dev_src_file, dev_trg_file))
test_data = list(read(test_src_file, test_trg_file))

unk_src = w2i_src['<unk>']
eos_src = w2i_src['</s>']
pad_src = w2i_src['<pad>']
w2i_src = defaultdict(lambda: unk_src, w2i_src)
unk_trg = w2i_trg["<unk>"]
eos_trg = w2i_trg['</s>']
sos_trg = w2i_trg['<s>']
pad_trg = w2i_trg['<pad>']
w2i_trg = defaultdict(lambda: unk_trg, w2i_trg)
i2w_trg = {v: k for k, v in w2i_trg.items()}

# Dataset class
class TranslationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, trg = self.data[idx]
        return torch.tensor(src), torch.tensor(trg)

# Collate function for DataLoader
def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, padding_value=pad_src, batch_first=True)
    trg_batch = pad_sequence(trg_batch, padding_value=pad_trg, batch_first=True)
    return src_batch, trg_batch

# Create DataLoader
BATCH_SIZE = 16
train_loader = DataLoader(TranslationDataset(train_data), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(TranslationDataset(dev_data), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

class Seq2Seq(nn.Module):
    def __init__(self, nwords_src, nwords_trg, embed_size, hidden_size):
        super(Seq2Seq, self).__init__()
        self.embedding_src = nn.Embedding(nwords_src, embed_size)
        self.embedding_trg = nn.Embedding(nwords_trg, embed_size)
        self.encoder_lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.decoder_lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, nwords_trg)
    
    def forward(self, src, trg):
        embedded_src = self.embedding_src(src)
        encoder_outputs, (hidden, cell) = self.encoder_lstm(embedded_src)

        # Decoder
        embedded_trg = self.embedding_trg(trg)
        decoder_outputs, _ = self.decoder_lstm(embedded_trg, (hidden, cell))
        logits = self.fc(decoder_outputs)
        return logits
    
    def generate(self, src, sos_trg, eos_trg, max_len):
        embedded_src = self.embedding_src(src)
        encoder_outputs, (hidden, cell) = self.encoder_lstm(embedded_src)

        # Initialize with <s> (start-of-sequence) token
        trg_indexes = [sos_trg]
        trg_sentence = []

        for _ in range(max_len):
            trg_tensor = torch.LongTensor([trg_indexes[-1]]).unsqueeze(0).to(src.device)  # Shape (1, 1)
            embedded_trg = self.embedding_trg(trg_tensor)
            decoder_output, (hidden, cell) = self.decoder_lstm(embedded_trg, (hidden, cell))
            output = self.fc(decoder_output.squeeze(0))
            next_word = output.argmax(1).item()

            if next_word == eos_trg:
                break
            trg_indexes.append(next_word)
            trg_sentence.append(i2w_trg[next_word])

        return trg_sentence

# Model and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Seq2Seq(len(w2i_src), len(w2i_trg), 64, 128).to(device)
optimizer = optim.Adam(model.parameters())

# Ignore padding token in loss calculation
criterion = nn.CrossEntropyLoss(ignore_index=pad_trg)

# Training and evaluation functions
def train_epoch(model, train_loader, optimizer):
    model.train()
    total_loss = 0
    total_words = 0

    for src_batch, trg_batch in tqdm(train_loader):
        src_batch, trg_batch = src_batch.to(device), trg_batch.to(device)
        trg_input = trg_batch[:, :-1]
        trg_output = trg_batch[:, 1:]

        outputs = model(src_batch, trg_input)
        outputs = outputs.view(-1, outputs.shape[-1])

        # Only count non-padding tokens
        non_pad_elements = trg_output.ne(pad_trg).sum().item()

        loss = criterion(outputs, trg_output.contiguous().view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * non_pad_elements  # multiply by non-padded tokens
        total_words += non_pad_elements

    return total_loss / total_words

def evaluate(model, dev_loader):
    model.eval()
    total_loss = 0
    total_words = 0

    with torch.no_grad():
        for src_batch, trg_batch in tqdm(dev_loader):
            src_batch, trg_batch = src_batch.to(device), trg_batch.to(device)
            trg_input = trg_batch[:, :-1]
            trg_output = trg_batch[:, 1:]

            outputs = model(src_batch, trg_input)
            outputs = outputs.view(-1, outputs.shape[-1])

            # Only count non-padding tokens
            non_pad_elements = trg_output.ne(eos_trg).sum().item()

            loss = criterion(outputs, trg_output.contiguous().view(-1))

            total_loss += loss.item() * non_pad_elements  # multiply by non-padded tokens
            total_words += non_pad_elements

    return total_loss / total_words

# Training loop
max_test_bleu = 0.0
for epoch in range(100):
    train_loss = train_epoch(model, train_loader, optimizer)
    dev_loss = evaluate(model, dev_loader)
    print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Dev Loss: {dev_loss:.4f}, Perplexity: {math.exp(dev_loss):.4f}")

    # Generate translation
    test_loader = DataLoader(TranslationDataset(test_data), batch_size=1, shuffle=False, collate_fn=collate_fn)
    bleu_scores = []
    for src_batch, trg_batch in test_loader:
        src_batch = src_batch.to(device)
        translated_sent = model.generate(src_batch[0].unsqueeze(0), sos_trg, eos_trg, 50)  # Process single sentence
        translated_sent = ' '.join(translated_sent)
        
        # Convert target sentence to words using i2w_trg
        reference_sent = [i2w_trg[token.item()] for token in trg_batch[0] if token.item() != pad_trg]
        reference_sent = reference_sent[1:-1]  # Remove <s> and </s>
        
        # Calculate BLEU score
        bleu_score = sentence_bleu([reference_sent], translated_sent.split(), weights=(0.5, 0.5, 0, 0))
        bleu_scores.append(bleu_score)
        
        # print(f"Reference: {' '.join(reference_sent)}")
        # print(f"Translated: {translated_sent}")
        # print(f"BLEU Score: {bleu_score:.4f}\n")
            
    # Average BLEU score across the test set
    test_bleu = sum(bleu_scores) / len(bleu_scores)

    if max_test_bleu < test_bleu:
        max_test_bleu = test_bleu
    print(f"epoch {epoch}: test acc={test_bleu:.4f}")
    
print("max test bleu=%.4f" % (max_test_bleu))