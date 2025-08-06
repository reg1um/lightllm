import numpy as np
import pandas as pd
import os.path
import time
import random as rd

# For Word Tokenization
import re
from collections import Counter

from pretraitment import tokenize, untokenize
from datasets import load_dataset

import torch
import torch.nn as nn


# CONSTANTS
CONTEXT_LENGTH = 256
BATCH_SIZE = 64
D_MODEL = 384
NB_HEADS = 6
NB_EPOCH = 300
LR = 3e-4
NB_LAYERS = 6
NB_TOKENS = 5000
DATA_DIR = "data/"
DROPOUT = 0.2
TOKENIZE_METHOD = "char"  # 'char' or 'word'
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


    # Stream usage for word tokenization in order not to load the entire file into memory
    def word_stream(line):
        for match in re.finditer(r'\b\w+\b', line.lower()):
            yield match.group(0)

    def build_vocab(filepath, vocab_size=10000):
        counter = Counter()
        biggest_word = 0

        for word in word_stream(filepath):
            counter[word] += 1
            biggest_word = max(biggest_word, len(word))

        # Top N-1 words, +1 for <unk>
        vocab = {word: idx for idx, (word, _) in enumerate(counter.most_common(vocab_size - 1))}
        vocab['<unk>'] = vocab_size - 1
        rev_vocab = {idx: word for word, idx in vocab.items()}
        if len(vocab) < vocab_size:
            print(f"Error: Vocabulary size ({len(vocab)}) is less than requested size ({vocab_size}).")
            return
        return vocab, rev_vocab, biggest_word

    def tokenize_words(text, vocab):
        tokens = []
        for word in word_stream(text):
            tokens.append(vocab.get(word, vocab['<unk>']))
            if len(tokens) >= CONTEXT_LENGTH+1:
                break
        return tokens

    def untokenize_words(tokens, rev_vocab):
        untokenized = []
        for token in tokens:
            untokenized.append(rev_vocab.get(token, '<unk>'))
        return ' '.join(untokenized)

    def get_words(text, clen=CONTEXT_LENGTH+1):
        words = ""
        current_word = ""
        for char in text:
            if char.isalnum() or char == "'":
                current_word += char
            else:
                if current_word != "":
                    words += " " + current_word
                    current_word = ""
                if len(words.split()) >= clen:
                    break
        if current_word:
            words += " " + current_word
        return words

    def get_batches(text, context_length=CONTEXT_LENGTH, batch_size=BATCH_SIZE, biggest_word=0):
        batches = []
        targets = []

        while len(batches) < batch_size:
            # Select a random starting point in the training text

            if TOKENIZE_METHOD == "char":
                rand = np.random.randint(0, len(text) - context_length - 1)
                chunk = text[rand:rand + context_length + 1]
                chunk = tokenize(chunk, vocabulary)

            # The challenge for 'word' is that we need the correct amound of words
            # We have to make sure that we're not too close to the end of the text
            elif TOKENIZE_METHOD == "word":
                # TODO: Check if that's really necessary or just loop until we get a valid chunk
                rand = np.random.randint(0, len(text) - (context_length) * (biggest_word + 1))
                chunk = get_words(text[rand:], clen=context_length + 1)
                chunk = tokenize_words(chunk, vocabulary)

                # If the chunk is too short, we skip it
                if len(chunk) < context_length + 1:
                    continue

            input_seq = chunk[:context_length]
            target_seq = chunk[1:context_length + 1]
            batches.append(input_seq)
            targets.append(target_seq)

        return batches, targets

    # ============================================= Pretraitment =============================================


    # Concatenate all stories to a single big file
    # If already done, then load the files into variables


    test = ""
    train = ""
    if not os.path.exists(DATA_DIR + "train.txt"):
        for stories in ds['train']:
            train += stories['text']

        with open(DATA_DIR + "train.txt", 'w') as train_file:
            train_file.write(train)
        print("Stories stored in \"train.txt\"")
    else:
        with open(DATA_DIR + "train.txt", 'r') as train_file:
            train = train_file.read()
            print("train.txt loaded!")

    if not os.path.exists(DATA_DIR + "test.txt"):
        for stories in ds['validation']:
            test += stories['text']
        with open(DATA_DIR + "test.txt", 'w') as test_file:
            test_file.write(test)
        print("Stories stored in \"test.txt\"")
    else:
        with open(DATA_DIR + "test.txt", 'r') as test_file:
            test = test_file.read()
            print("test.txt loaded!")

    #train = train[:100000]  # For testing purposes, limit the training text size

    # Define the Dataset Vocabulary, used for tokenization
    biggest_word = 0
    if TOKENIZE_METHOD == "char":
        vocabulary = {letter:idx for idx,letter in enumerate(sorted(list(set(test))))}
        rev_vocabulary = {idx:letter for idx,letter in enumerate(sorted(list(set(test))))}

        print(f"Vocabulary: {vocabulary}")

        # To test with a smaller file
        # tmp_train = train[:10000]

        # Tokenize the train text
        print(f"Tokenizing the training text")
        tokenized = tokenize(train, vocabulary)
        print(f"Text Tokenized !")
    elif TOKENIZE_METHOD == "word":


        print(f"Counting words in the training text")
        vocabulary, rev_vocabulary, biggest_word = build_vocab(train, vocab_size=10000)
        print(f"Vocabulary built with {len(vocabulary)} words")

        """
        print(f"Tokenizing the training text")
        tokenized = list(token_stream("train.txt", vocabulary))
        print(f"Text Tokenized !")
        print(tokenized[:100])  # Display the first 100 tokens for verification
        """
    print(f"Biggest word length: {biggest_word}")


    # ============================================= Manage Training Batches =============================================

    """
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
    """

    # ============================================= Embedding =============================================

    vocab_size = len(vocabulary)
    print(f"Vocabulary size: {vocab_size}")

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
        batches, targets = get_batches(train, biggest_word=biggest_word)

        for b in batches:
            for token in b:
                if token >= vocab_size or token < 0:
                    print(f"Invalid token index: {token}, vocab size: {vocab_size}")

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

    # Start of the generation
    context = "This"
    context_tokens = torch.tensor([tokenize_words(context, vocabulary)]).to(device)

    model.eval()
    output_tokens = model.generate(context_tokens)[0].tolist()

    if TOKENIZE_METHOD == "char":
        generated_text = untokenize(output_tokens, rev_vocabulary)
    elif TOKENIZE_METHOD == "word":
        generated_text = untokenize_words(output_tokens, rev_vocabulary)

    print(generated_text)
    print(f"\nGenerated text length: {len(generated_text)} characters")
    end_time = time.time()
    print(f"Training and Generation completed in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
