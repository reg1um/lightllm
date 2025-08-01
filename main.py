import numpy as np
import pandas as pd
import os.path
import time
import random as rd

from pretraitment import tokenize, untokenize
from datasets import load_dataset

import torch
import torch.nn as nn


# CONSTANTS
CONTEXT_LENGTH = 256
BATCH_SIZE = 64
D_MODEL = 384
NB_HEADS = 6
NB_EPOCH = 4000
LR = 3e-4
NB_LAYERS = 6
NB_TOKENS = 2000
DROPOUT = 0.2
device='cuda'
rd.seed(42)


def main():

    ds = load_dataset("roneneldan/TinyStories")


    # ============================================= Utils =============================================
    def tokenize(text, vocabulary):
        tokenized = []
        for letter in text:
            tokenized.append(vocabulary[letter])
        return tokenized

    def untokenize(text, rev_vocabulary):
        untokenized = ""
        for tokens in text:
            untokenized += rev_vocabulary[tokens]
        return untokenized

    # ============================================= Pretraitment =============================================


    # Concatenate all stories to a single big file
    # If already done, then load the files into variables

    test = ""
    train = ""
    if not os.path.exists("train.txt"):
        for stories in ds['train']:
            train += stories['text']

        with open("train.txt", 'w') as train_file:
            train_file.write(train)
        print("Stories stored in \"train.txt\"")
    else:
        with open("train.txt", 'r') as train_file:
            train = train_file.read()
            print("train.txt loaded!")

    if not os.path.exists("test.txt"):
        for stories in ds['validation']:
            test += stories['text']
        with open("test.txt", 'w') as test_file:
            test_file.write(test)
        print("Stories stored in \"test.txt\"")
    else:
        with open("test.txt", 'r') as test_file:
            test = test_file.read()
            print("test.txt loaded!")

    # Define the Dataset Vocabulary, used for tokenization
    vocabulary = {letter:idx for idx,letter in enumerate(sorted(list(set(test))))}
    rev_vocabulary = {idx:letter for idx,letter in enumerate(sorted(list(set(test))))}

    print(f"Vocabulary: {vocabulary}")

    # To test with a smaller file
    #tmp_train = train[:10000]

    # Tokenize the train text
    print(f"Tokenizing the training text")
    tokenized = tokenize(train, vocabulary)
    print(f"Text Tokenized !")


    # ============================================= Manage Training Batches =============================================

    # Create batches of randomly selected data
    def get_batches():
        batches = []
        targets = []

        while len(batches) < BATCH_SIZE:
            rand = np.random.randint(0, len(tokenized) - CONTEXT_LENGTH - 1)
            chunk = tokenized[rand:rand + CONTEXT_LENGTH + 1]
            input_seq = chunk[:CONTEXT_LENGTH]
            target_seq = chunk[1:]
            batches.append(input_seq)
            targets.append(target_seq)

        return batches, targets

    # ============================================= Embedding =============================================

    vocab_size = len(vocabulary)

    # Embedding every characters
    class CharEmbedding(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_embedding_table = nn.Embedding(vocab_size, D_MODEL)
            self.pos_embedding_table = nn.Embedding(CONTEXT_LENGTH, D_MODEL)

        def forward(self, input):
            pos = torch.arange(input.shape[1], device=input.device)
            tok_embedding = self.token_embedding_table(input)
            pos_embedding = self.pos_embedding_table(pos)[None, :, :]  # To have a (1, Context len, D_model len) shape
            return tok_embedding + pos_embedding


    # ============================================= Transformer =============================================

    class FeedForward(nn.Module):
        def __init__(self, n_embed):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_embed, 4 * n_embed), 
                nn.ReLU(),
                nn.Linear(4 * n_embed, n_embed),
                nn.Dropout(DROPOUT)
            )

        def forward(self, x):
            return self.net(x)


    class AttentionHead(nn.Module):
        def __init__(self, head_size):
            super().__init__()
            self.query = nn.Linear(D_MODEL, head_size, bias=False)
            self.key = nn.Linear(D_MODEL, head_size, bias=False)
            self.value = nn.Linear(D_MODEL, head_size, bias=False)
            self.dropout = nn.Dropout(DROPOUT)
            self.register_buffer("mask", torch.tril(torch.ones(CONTEXT_LENGTH + 1, CONTEXT_LENGTH + 1)))

        def forward(self, x):
            batch, pos, channel = x.shape

            k = self.key(x)
            q = self.query(x)
            v = self.value(x)

            # Dot Product Attention
            weights = q @ k.transpose(-2, -1) * channel ** -0.5

            # Eliminate "future" characters so that we don't interact with them
            weights = weights.masked_fill(self.mask[:pos, :pos] == 0, float('-inf'))

            # Turn Scores into proper Weights
            weights = torch.softmax(weights, dim=-1)

            weights = self.dropout(weights)

            output = weights @ v
            return output

    class AttentionMultiHead(nn.Module):
        def __init__(self, head_size, nb_heads):
            super().__init__()
            self.heads = nn.ModuleList([AttentionHead(head_size) for _ in range(nb_heads)])
            self.proj = nn.Linear(D_MODEL, D_MODEL)
            self.dropout = nn.Dropout(DROPOUT)

        def forward(self, x):
            head_outputs = [h(x) for h in self.heads]
            out = torch.cat(head_outputs, dim=-1)
            out = self.proj(out)
            out = self.dropout(out)
            return out

    class TransformerBlock(nn.Module):
        def __init__(self, nb_embed, nb_heads):
            super().__init__()
            head_size = nb_embed // nb_heads
            self.heads = AttentionMultiHead(head_size, nb_heads)
            self.ffwd = FeedForward(nb_embed)
            self.ln1 = nn.LayerNorm(nb_embed)
            self.ln2 = nn.LayerNorm(nb_embed)

        def forward(self, x):
            # Residual Connections
            x = x + self.heads(self.ln1(x))
            x = x + self.ffwd(self.ln2(x))
            return x

    class LanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embd = CharEmbedding()
            self.trans = nn.ModuleList([TransformerBlock(D_MODEL, NB_HEADS) for _ in range(NB_LAYERS)])
            self.ln = nn.LayerNorm(D_MODEL)
            self.ffwd = nn.Linear(D_MODEL, vocab_size)

        def forward(self, x):
            x = self.embd(x)
            for block in self.trans:
                x = block(x)
            x = self.ln(x)
            x = self.ffwd(x)
            return x

        @torch.no_grad()
        def generate(self, start):
            for _ in range(NB_TOKENS):
                # Ensure the context has the correct length
                if start.shape[1] < CONTEXT_LENGTH:
                    # Pad on the left with zeros (or any special token)
                    padding = torch.zeros((1, CONTEXT_LENGTH - start.shape[1]), dtype=torch.long, device=start.device)
                    context = torch.cat([padding, start], dim=1)
                else:
                    context = start[:, -CONTEXT_LENGTH:]

                logits = self.forward(context)
                last = logits[:, -1, :]
                probs = torch.softmax(last, dim=-1)

                next_token = torch.multinomial(probs, num_samples=1)
                start = torch.cat([start, next_token], dim=1)

            return start


    # ============================================= Training Loop =============================================

    loss_fn = nn.CrossEntropyLoss()
    model = LanguageModel()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    start_time = time.time()

    for i in range(NB_EPOCH):
        batches, targets = get_batches()

        batches = torch.LongTensor(batches).to(device)
        targets = torch.LongTensor(targets).to(device)

        logits = model(batches)
        loss = loss_fn(logits.view(-1, vocab_size), targets.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f"Step {i}: Loss = {loss.item():.4f}")


    # ============================================= Generation =============================================

    context = " "
    context_tokens = torch.tensor([tokenize(context, vocabulary)]).to(device)

    model.eval()
    output_tokens = model.generate(context_tokens)[0].tolist()

    generated_text = untokenize(output_tokens, rev_vocabulary)

    print(generated_text)
    print(f"Generated text length: {len(generated_text)} characters")
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")






if __name__ == "__main__":
    main()
