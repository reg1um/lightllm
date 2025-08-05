# LightLLM

A personal learning project to understand how transformers work from the ground up. Built while studying the famous "Attention is All You Need" paper and other transformer resources.

## This project

This is my attempt at implementing a mini GPT-style language model from scratch. The goal was to really understand how attention mechanisms and transformers work, rather than just using pre-built libraries. It's trained on the "simple stories" dataset and can generate new text character by character in its style.

## What I learned

- How self-attention actually works under the hood
- Why positional embeddings matter
- How to stack transformer blocks
- The importance of residual connections and layer normalization
- How to generate text autoregressive

## How to run it

Make sure you have the basics installed:
```bash
pip install torch numpy pandas datasets
```

Then just run:
```bash
python main.py
```

It will download some story data, train a small model, and show you what kind of text it can generate. The whole process takes a few minutes depending on your hardware.

## What's in here

- `main.py` - Everything! The model, training loop, and text generation. I might factorize that in the future to make it more modular and readable.

## The model

It's intentionally small and simple:
- **Tokenization**: Supports both character-level and word-level tokenization (configurable via `TOKENIZE_METHOD`)
  - Character-level: Learns from individual character patterns
  - Word-level: Uses a vocabulary of the 10,000 most common words (configurable) with `<unk>` token for out-of-vocabulary words
- Has 6 transformer layers with 6 attention heads each
- Context window of 256 tokens (characters or words depending on tokenization method)
- About 384-dimensional embeddings (64 per head)

The idea was to keep it simple enough to understand every piece, while still being complex enough to generate somewhat coherent text.

## What's next
- Optimize the hyperparameters for better performance
- Benchmark it against other small publicly available models
- Experiment with different vocabulary sizes and tokenization strategies
- Many other improvements and experiments!

## Inspiration

Built following the "Attention is All You Need" paper and various online tutorials about transformers, from Karpathy to 3Blue1Brown videos. This was purely a learning exercise to demystify how these models actually work, without using AI to guide me!
