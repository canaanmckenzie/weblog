---
title: "The Architecture of Paying Attention"
subtitle: "Part 3 of 4 · Putting all the pieces together"
description: "A complete walkthrough of the transformer architecture, from embeddings to output."
date: 2025-12-28
template: post
uses_math: true
uses_code: true
---

If self-attention is the engine of the transformer, as we established last time, then what are the other parts? What's the chassis, the transmission, the inexplicable dashboard light that's been on for three months[^1]? Let's zoom out and see the whole machine.

## The 30,000-Foot View

A transformer for language modeling takes a sequence of tokens and produces, for each position, a probability distribution over what token should come next. The architecture has roughly these components:

1. **Token Embeddings**: Convert discrete tokens to continuous vectors
2. **Positional Encodings**: Inject information about position in the sequence
3. **Transformer Blocks** (repeated N times):
   - Multi-head self-attention
   - Feed-forward network
   - Layer normalization
   - Residual connections
4. **Output Projection**: Convert vectors back to vocabulary probabilities

Let's go through each piece.

## Token Embeddings

Tokens are discrete things—integers, basically—and neural networks need continuous things they can differentiate through. So we maintain a learned embedding matrix $E$ of shape $(\text{vocab\_size}, d_{\text{model}})$. To convert a token ID to a vector, we just look up its row:

```python
class TokenEmbedding:
    def __init__(self, vocab_size, d_model):
        # Random initialization; learned during training
        self.embedding = np.random.randn(vocab_size, d_model) * 0.02
    
    def __call__(self, token_ids):
        """
        token_ids: array of shape (seq_len,) with integer token IDs
        returns: array of shape (seq_len, d_model)
        """
        return self.embedding[token_ids]
```

Simple, right? Each row of the embedding matrix is, in some sense, what the model "thinks" about that token.

## Positional Encodings: The Problem of Order

Here's an issue: self-attention, as we defined it, is completely permutation-invariant. If you shuffle the input sequence, you get the same output (also shuffled). But order matters in language! "The dog bit the man" and "The man bit the dog" are very different sentences[^2].

The solution is to add positional information to the embeddings. The original transformer used sinusoidal functions:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

For each position $pos$ and each dimension $i$, we compute a sine or cosine with a specific frequency. The frequencies decrease geometrically with the dimension index, creating a kind of binary-ish encoding where low dimensions vary rapidly with position and high dimensions vary slowly.

```python
def positional_encoding(seq_len, d_model):
    """
    Generate sinusoidal positional encodings.
    """
    positions = np.arange(seq_len)[:, np.newaxis]  # (seq_len, 1)
    dims = np.arange(d_model)[np.newaxis, :]       # (1, d_model)
    
    # Frequency for each dimension
    angles = positions / np.power(10000, (2 * (dims // 2)) / d_model)
    
    # Apply sin to even indices, cos to odd indices
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    
    return pe
```

Modern models like GPT often just learn the positional embeddings directly—another embedding matrix, this time indexed by position. Both approaches work.

## The Transformer Block

The real meat is in the repeated transformer blocks. Each block has two sub-layers: multi-head attention and a feed-forward network. Both use residual connections and layer normalization.

### Residual Connections

A residual connection is just: $\text{output} = x + f(x)$. Instead of learning the full transformation, we learn the *difference* (residual) from the input. This helps with gradient flow during training and makes it possible to stack many layers without things degrading.

### Layer Normalization

Layer norm normalizes each sample independently across its features:

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sigma + \epsilon} + \beta$$

where $\mu$ and $\sigma$ are the mean and standard deviation computed across the feature dimension, and $\gamma$, $\beta$ are learned scale and shift parameters.

```python
class LayerNorm:
    def __init__(self, d_model, eps=1e-6):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps
    
    def __call__(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
```

### Feed-Forward Network

Between attention layers sits a simple two-layer MLP applied to each position independently:

$$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$

The inner dimension is typically 4× the model dimension (so if $d_{\text{model}} = 512$, the inner dimension is 2048). This is where a lot of the model's "knowledge" gets stored during training[^3].

```python
class FeedForward:
    def __init__(self, d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff) * 0.02
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.02
        self.b2 = np.zeros(d_model)
    
    def __call__(self, x):
        hidden = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        return hidden @ self.W2 + self.b2
```

### Putting It Together: One Block

A single transformer block combines everything:

```python
class TransformerBlock:
    def __init__(self, d_model, n_heads, d_ff):
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
    
    def __call__(self, x):
        # Attention with residual and layer norm
        attn_out = self.attention(x)
        x = self.ln1(x + attn_out)
        
        # FFN with residual and layer norm
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)
        
        return x
```

Note the order: attention → add residual → normalize → FFN → add residual → normalize. This is called "Post-LN." Some models use "Pre-LN" which puts the normalization before each sub-layer. Both work; the debates about which is better are ongoing[^4].

## Stacking Blocks

The full transformer is just a stack of these blocks. GPT-2 small uses 12 blocks. GPT-3 uses 96. Each block refines the representations, allowing progressively more abstract features to emerge.

```python
class Transformer:
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_seq_len):
        self.token_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_enc = positional_encoding(max_seq_len, d_model)
        self.blocks = [
            TransformerBlock(d_model, n_heads, d_ff) 
            for _ in range(n_layers)
        ]
        self.output_proj = np.random.randn(d_model, vocab_size) * 0.02
    
    def __call__(self, token_ids):
        seq_len = len(token_ids)
        
        # Embed tokens and add positional information
        x = self.token_emb(token_ids) + self.pos_enc[:seq_len]
        
        # Pass through all blocks
        for block in self.blocks:
            x = block(x)
        
        # Project to vocabulary logits
        logits = x @ self.output_proj
        return logits
```

## Causal Masking: The Decoder's Secret

One thing we haven't discussed: for language modeling (predicting the next token), we need to prevent the model from "cheating" by looking at future tokens. The solution is causal masking—we modify the attention scores to set future positions to $-\infty$ before the softmax:

```python
def causal_mask(seq_len):
    """Lower triangular mask for causal attention."""
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    mask[mask == 1] = -np.inf
    return mask

# In attention: scores = scores + causal_mask(seq_len)
```

This ensures that position $i$ can only attend to positions $\leq i$. The model learns to predict token $i+1$ from tokens $1 \ldots i$.

## What Have We Built?

We now have all the pieces of a decoder-only transformer (like GPT):

- Token embeddings to convert IDs to vectors
- Positional encodings to preserve order information
- Multi-head attention to mix information across positions
- Feed-forward networks to transform representations
- Layer normalization and residual connections for stable training
- Causal masking to prevent future-peeking

In the final part, we'll put it all together into a complete, runnable implementation—a toy language model that can actually learn to predict characters. The code won't be efficient, but it'll be clear.

[^1]: The check engine light of deep learning is the loss curve that starts out looking great and then mysteriously plateaus or, worse, starts climbing. When this happens you check: the learning rate, the batch size, the data preprocessing, the initialization scheme, whether you accidentally left dropout at 0.9 instead of 0.1, etc. Often the answer is embarrassingly simple. Sometimes it's not.

[^2]: Though both describe regrettable incidents of biting.

[^3]: There's been interesting research suggesting that the FFN layers act as a kind of key-value memory. Each row of $W_1$ is a "key" that matches certain input patterns, and the corresponding row of $W_2$ is the "value" that gets added to the output. This gives some insight into how factual knowledge gets encoded (and why it's so hard to reliably edit or remove).

[^4]: Pre-LN tends to be more stable during training, allowing for higher learning rates. Post-LN (the original formulation) can achieve slightly better final performance with careful tuning. The fact that both work, and the reasons why are subtle, tells you something about how much of deep learning is still empirical alchemy.
