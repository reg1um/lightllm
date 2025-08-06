# Implementation of a GPT-2 Tiktoken-like tokenizer from scratch

def bpe_freq(tokens):
    # Computes the frequency of pairs of tokens in the list
    counter = {}
    for i in zip(tokens, tokens[1:]):
        pair = (i[0], i[1])
        counter[pair] = counter.get(pair, 0) + 1
    return counter

def bpe_reduce(tokens, pairs, idx):
    # Reduces the tokens by replacing pairs with their corresponding indices
    new_tokens = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pairs:
            new_tokens.append(idx)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens

def bpe_merge(tokens, num_merges):
    # Create vocabulary using BPE ("train the tokenizer")
    merges = {}
    for i in range(num_merges):
        top_pairs = sorted(bpe_freq(tokens).items(), key=lambda x: x[1], reverse=True)[:num_merges]
        idx = 256 + i
        pair = top_pairs[i][0]
        print(f"Merging pair {pair} into index {idx}")
        tokens = bpe_reduce(tokens, pair, idx)
        merges[pair] = idx
    return merges, tokens


def decode(tokens, vocab):
    # Decodes a list of tokens back to text using the vocabulary
    b = b"".join([vocab[idx] for idx in tokens])
    text = b.decode('utf-8', errors='replace')
    return text

def encode(text, merges):
    # Encodes a text string into tokens using the BPE merges
    tokens = list(text.encode("utf-8"))
    while True:
        freq = bpe_freq(tokens)
        # Get a pair in freq that is one of the "first" in merges
        pair = min(freq, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges: # There is nothing to merge :<
            break
        idx = merges[pair]
        tokens = bpe_reduce(tokens, pair, idx)
    return tokens



def main():
    text = "Hello, world! This is a test."
    lorem_ipsum = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
    tokens = list(map(int, lorem_ipsum.encode('utf-8')))

    print(f"Initial tokens: {tokens}")
    print(f"Number of tokens: {len(tokens)}")


    num_merges = 30
    merges, tokens = bpe_merge(tokens, num_merges)

    print(f"Merges: {merges}")
    
    vocab = {idx: bytes([idx]) for idx in range(256)}

    for (a, b), idx in merges.items():
        vocab[idx] = vocab[a] + vocab[b]
    print(vocab)

    encoded = encode(lorem_ipsum, merges)
    print(f"Encoded text is : {encoded}")
    print(f"Number of encoded tokens: {len(encoded)}")
    print(f"Decoded text is : {decode(encoded, vocab)}")
    print(f"Compression ratio: {len(lorem_ipsum) / len(encoded)}")

    #print(sorted(((v, k) for k, v in bpe_freq(tokens).items()), reverse=True))

if __name__ == "__main__":
    main()
