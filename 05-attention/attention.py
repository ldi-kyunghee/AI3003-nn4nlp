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
from plot_attention import plot_attention
import copy

# Data paths
train_src_file = "../data/parallel/train.ja"
train_trg_file = "../data/parallel/train.en"
dev_src_file = "../data/parallel/dev.ja"
dev_trg_file = "../data/parallel/dev.en"
test_src_file = "../data/parallel/test.ja"
test_trg_file = "../data/parallel/test.en"

w2i_src = defaultdict(lambda: len(w2i_src))
w2i_trg = defaultdict(lambda: len(w2i_trg))

# Hyperparameters
EMBED_SIZE = 64
HIDDEN_SIZE = 128
ATTENTION_SIZE = 128
BATCH_SIZE = 16
MAX_SENT_SIZE = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Read parallel data
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
unk_trg = w2i_trg['<unk>']
eos_trg = w2i_trg['</s>']
sos_trg = w2i_trg['<s>']
pad_trg = w2i_trg['<pad>']
w2i_trg = defaultdict(lambda: unk_trg, w2i_trg)
i2w_trg = {v: k for k, v in w2i_trg.items()}
i2w_src = {v: k for k, v in w2i_src.items()}

nwords_src = len(w2i_src)
nwords_trg = len(w2i_trg)

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
train_loader = DataLoader(TranslationDataset(train_data), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(TranslationDataset(dev_data), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

class Seq2SeqAttention(nn.Module):
    def __init__(self, nwords_src, nwords_trg, embed_size, hidden_size, attention_size):
        super(Seq2SeqAttention, self).__init__()
        self.embedding_src = nn.Embedding(nwords_src, embed_size)
        self.embedding_trg = nn.Embedding(nwords_trg, embed_size)
        self.encoder_lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.decoder_lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, nwords_trg)
    
        # Attention parameters
        self.att_w1_src = nn.Linear(hidden_size, attention_size)
        self.att_w1_tgt = nn.Linear(hidden_size, attention_size)
        self.att_w2 = nn.Linear(attention_size, 1)
        
        # Output layers
        self.out_w = nn.Linear(hidden_size*2, hidden_size)
        self.out_sm = nn.Linear(hidden_size, nwords_trg)

    def calc_attention(self, src_vectors, tgt_vector):
        att_src = self.att_w1_src(src_vectors)
        att_tgt = self.att_w1_tgt(tgt_vector).unsqueeze(1)
        att_combined = torch.tanh(att_src + att_tgt)
        attention_scores = self.att_w2(att_combined).squeeze(2)
        alignment = torch.softmax(attention_scores, dim=1)
        att_vector = torch.bmm(alignment.unsqueeze(1), src_vectors).squeeze(1)
        return att_vector, alignment

    def forward(self, src, trg):
        embedded_src = self.embedding_src(src)
        encoder_outputs, _ = self.encoder_lstm(embedded_src)

        # Decoder
        embedded_trg = self.embedding_trg(trg)
        
        # Decoder LSTM and attention
        all_logits = []
        trg_len = trg.size(1)
        batch_size = src.size(0)
        
        h, c = self.init_hidden(batch_size)
        for i in range(trg_len - 1):
            tgt_input = embedded_trg[:, i, :].unsqueeze(1)
            lstm_output, (h, c) = self.decoder_lstm(tgt_input, (h, c))
            att_output, _ = self.calc_attention(encoder_outputs, lstm_output.squeeze(1))
            concat_output = torch.cat([lstm_output.squeeze(1), att_output], dim=1)
            final_output = torch.tanh(self.out_w(concat_output))
            logits = self.out_sm(final_output)
            all_logits.append(logits)

        return torch.stack(all_logits, dim=1)  # Shape: (batch_size, trg_len-1, vocab_size)

    def generate(self, src_sent):
        # Embed source sentence
        src_embedded = self.embedding_src(src_sent)  # Shape: (batch_size, seq_len, embed_size)
        
        # Pass through encoder LSTM
        src_outputs, (hidden, cell) = self.encoder_lstm(src_embedded)  # Shape: (batch_size, seq_len, hidden_size * 2)
        
        # Initialize decoder hidden and cell state
        trg_sent = []
        prev_word = torch.tensor([sos_trg], device=device)
        attention_matrix = []
        
        for _ in range(MAX_SENT_SIZE):
            # Embed the previous target word
            tgt_input = self.embedding_trg(prev_word).unsqueeze(1)  # Shape: (batch_size, 1, embed_size)
            
            # Decoder LSTM step
            lstm_output, (hidden, cell) = self.decoder_lstm(tgt_input, (hidden, cell))
            
            # Calculate attention
            att_output, alignment = self.calc_attention(src_outputs, lstm_output.squeeze(1))
            attention_matrix.append(alignment)
            
            # Concatenate LSTM output and attention output
            concat_output = torch.cat([lstm_output.squeeze(1), att_output], dim=1)
            
            # Final output layer
            final_output = torch.tanh(self.out_w(concat_output))
            logits = self.out_sm(final_output)
            next_word = torch.argmax(logits, dim=1).item()
            
            trg_sent.append(next_word)
            prev_word = torch.tensor([next_word], device=device)

            if next_word == eos_trg:
                break
        
        return [i2w_trg[word] for word in trg_sent], torch.stack(attention_matrix)

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, HIDDEN_SIZE, device=device),
                torch.zeros(1, batch_size, HIDDEN_SIZE, device=device))


# Model and optimizer
model = Seq2SeqAttention(nwords_src, nwords_trg, EMBED_SIZE, HIDDEN_SIZE, ATTENTION_SIZE).to(device)
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
        trg_input = trg_batch[:, :]  # Input to the decoder
        trg_output = trg_batch[:, 1:]  # Expected output (excluding <s>)

        # Forward pass
        outputs = model(src_batch, trg_input)  # (batch_size, trg_len-1, vocab_size)
        
        # Flatten outputs to (batch_size * trg_len-1, vocab_size)
        outputs = outputs.view(-1, outputs.shape[-1])

        # Flatten trg_output to (batch_size * trg_len-1)
        trg_output = trg_output.contiguous().view(-1)

        # Compute loss, ignoring padding index
        loss = criterion(outputs, trg_output)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute total loss
        non_pad_elements = trg_output.ne(pad_trg).sum().item()  # Count non-padding tokens
        total_loss += loss.item() * non_pad_elements
        total_words += non_pad_elements

    return total_loss / total_words

def evaluate(model, dev_loader):
    model.eval()
    total_loss = 0
    total_words = 0

    with torch.no_grad():
        for src_batch, trg_batch in tqdm(dev_loader):
            src_batch, trg_batch = src_batch.to(device), trg_batch.to(device)
            trg_input = trg_batch[:, :]
            trg_output = trg_batch[:, 1:]

            outputs = model(src_batch, trg_input)  # (batch_size, trg_len-1, vocab_size)
            outputs = outputs.view(-1, outputs.shape[-1])  # Flatten for cross-entropy loss

            non_pad_elements = trg_output.ne(pad_trg).sum().item()

            loss = criterion(outputs, trg_output.contiguous().view(-1))  # Compute loss

            total_loss += loss.item() * non_pad_elements
            total_words += non_pad_elements

    return total_loss / total_words

# Training loop
max_test_bleu = -0.1
best_model = None
for epoch in range(100):
    train_loss = train_epoch(model, train_loader, optimizer)
    dev_loss = evaluate(model, dev_loader)
    print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Dev Loss: {dev_loss:.4f}, Perplexity: {math.exp(dev_loss):.4f}")

    # Generate translation and compute BLEU score on test set
    test_loader = DataLoader(TranslationDataset(test_data), batch_size=1, shuffle=False, collate_fn=collate_fn)
    bleu_scores = []
    for i, (src_batch, trg_batch) in enumerate(test_loader):
        src_batch = src_batch.to(device)
        translated_sent, attention_matrix = model.generate(src_batch[0].unsqueeze(0))
        translated_sent = ' '.join(translated_sent)
        
        reference_sent = [i2w_trg[token.item()] for token in trg_batch[0] if token.item() != pad_trg]
        reference_sent = reference_sent[1:-1]  # Remove <s> and </s>
        
        bleu_score = sentence_bleu([reference_sent], translated_sent.split(), weights=(0.5, 0.5, 0, 0))
        bleu_scores.append(bleu_score)

        if i <2:
            print(f"Reference: {' '.join(reference_sent)}")
            print(f"Translated: {translated_sent}")
            print(f"BLEU Score: {bleu_score:.4f}\n")
            
    # Average BLEU score across the test set
    test_bleu = sum(bleu_scores) / len(bleu_scores)

    if max_test_bleu < test_bleu:
        max_test_bleu = test_bleu
        best_model = copy.deepcopy(model.cpu())
    print(f"epoch {epoch}: test acc={test_bleu:.4f}")
    
print("max test bleu=%.4f" % (max_test_bleu))

# generate one sentence from test using the best model, and visualize it's attention
src = torch.tensor([test_data[0][0]]).to(device)
tgt = torch.tensor([test_data[0][1]]).to(device)
best_model = best_model.to(device)
input_sent = [i2w_src[x.item()] for x in src[0]]
output_sent, attention_matrix = best_model.generate(src)
attention_matrix = attention_matrix.squeeze(1).transpose(0,1).detach().cpu().numpy()

#note: this can break for long sentences with long words
plot_attention(input_sent, output_sent, attention_matrix, 'attention_matrix.png')