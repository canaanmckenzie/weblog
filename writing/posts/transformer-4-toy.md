---
title: "A Toy Example"
subtitle: "Part 4 of 4 · Learning to predict the next word, or: teaching a machine to dream in text"
description: "A complete, working transformer implementation that learns to predict characters."
date: 2025-12-28
template: post
uses_math: true
uses_code: true
---

We have now arrived at the part where we stop waving our hands and actually build the thing. All of it. A complete transformer that takes characters as input and learns to predict what character comes next[^1].

The code that follows is intentionally simple. It uses only NumPy. It is not fast. It will not win any benchmarks. But you can read it, understand it, and run it yourself. That's the point.

## The Complete Implementation

Here is a minimal, working character-level language model. I'll present it in sections, but it's a single coherent program.

### Imports and Helpers

```python
import numpy as np

def softmax(x, axis=-1):
    """Numerically stable softmax."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def cross_entropy_loss(logits, targets):
    """Cross-entropy loss for language modeling."""
    probs = softmax(logits)
    n = len(targets)
    log_probs = -np.log(probs[np.arange(n), targets] + 1e-9)
    return np.mean(log_probs)
```

### Multi-Head Attention

```python
class MultiHeadAttention:
    def __init__(self, d_model, n_heads):
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Projection matrices
        scale = 0.02
        self.W_Q = np.random.randn(d_model, d_model) * scale
        self.W_K = np.random.randn(d_model, d_model) * scale
        self.W_V = np.random.randn(d_model, d_model) * scale
        self.W_O = np.random.randn(d_model, d_model) * scale
    
    def __call__(self, x, mask=None):
        seq_len, d_model = x.shape
        
        # Project to Q, K, V
        Q = x @ self.W_Q
        K = x @ self.W_K
        V = x @ self.W_V
        
        # Reshape for multi-head: (seq_len, n_heads, d_head)
        Q = Q.reshape(seq_len, self.n_heads, self.d_head)
        K = K.reshape(seq_len, self.n_heads, self.d_head)
        V = V.reshape(seq_len, self.n_heads, self.d_head)
        
        # Compute attention for each head
        outputs = []
        for h in range(self.n_heads):
            q, k, v = Q[:, h, :], K[:, h, :], V[:, h, :]
            
            # Scaled dot-product attention
            scores = q @ k.T / np.sqrt(self.d_head)
            
            # Apply causal mask
            if mask is not None:
                scores = scores + mask
            
            weights = softmax(scores, axis=-1)
            out = weights @ v
            outputs.append(out)
        
        # Concatenate heads and project
        concat = np.concatenate(outputs, axis=-1)
        return concat @ self.W_O
```

### Feed-Forward and Layer Norm

```python
class FeedForward:
    def __init__(self, d_model, d_ff):
        scale = 0.02
        self.W1 = np.random.randn(d_model, d_ff) * scale
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * scale
        self.b2 = np.zeros(d_model)
    
    def __call__(self, x):
        hidden = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        return hidden @ self.W2 + self.b2

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

### Transformer Block

```python
class TransformerBlock:
    def __init__(self, d_model, n_heads, d_ff):
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
    
    def __call__(self, x, mask=None):
        # Attention with residual
        x = x + self.attn(self.ln1(x), mask)
        # FFN with residual
        x = x + self.ffn(self.ln2(x))
        return x
```

### The Complete Model

```python
class TinyTransformer:
    def __init__(self, vocab_size, d_model=64, n_heads=4, d_ff=256, 
                 n_layers=2, max_seq_len=128):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Embeddings
        scale = 0.02
        self.token_emb = np.random.randn(vocab_size, d_model) * scale
        self.pos_emb = np.random.randn(max_seq_len, d_model) * scale
        
        # Transformer blocks
        self.blocks = [
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ]
        
        # Output projection (tie with input embeddings for efficiency)
        self.output_proj = self.token_emb.T  # shape: (d_model, vocab_size)
    
    def __call__(self, token_ids):
        seq_len = len(token_ids)
        
        # Embed
        x = self.token_emb[token_ids] + self.pos_emb[:seq_len]
        
        # Causal mask
        mask = np.triu(np.full((seq_len, seq_len), -np.inf), k=1)
        
        # Forward through blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Project to logits
        logits = x @ self.output_proj
        return logits
```

### Training Loop

Now we need a way to train this thing. We'll use the simplest possible approach: numerical gradient estimation[^2]. This is absurdly slow but requires no backpropagation code.

```python
def get_all_params(model):
    """Collect all trainable parameters."""
    params = []
    params.append(('token_emb', model.token_emb))
    params.append(('pos_emb', model.pos_emb))
    for i, block in enumerate(model.blocks):
        params.append((f'block{i}.attn.W_Q', block.attn.W_Q))
        params.append((f'block{i}.attn.W_K', block.attn.W_K))
        params.append((f'block{i}.attn.W_V', block.attn.W_V))
        params.append((f'block{i}.attn.W_O', block.attn.W_O))
        params.append((f'block{i}.ffn.W1', block.ffn.W1))
        params.append((f'block{i}.ffn.b1', block.ffn.b1))
        params.append((f'block{i}.ffn.W2', block.ffn.W2))
        params.append((f'block{i}.ffn.b2', block.ffn.b2))
        params.append((f'block{i}.ln1.gamma', block.ln1.gamma))
        params.append((f'block{i}.ln1.beta', block.ln1.beta))
        params.append((f'block{i}.ln2.gamma', block.ln2.gamma))
        params.append((f'block{i}.ln2.beta', block.ln2.beta))
    return params

def train_step_numerical(model, x, y, lr=0.001, eps=1e-5):
    """One training step using numerical gradients (very slow!)."""
    params = get_all_params(model)
    base_loss = cross_entropy_loss(model(x), y)
    
    for name, param in params:
        flat = param.flatten()
        grad = np.zeros_like(flat)
        
        # Sample a subset of parameters for speed
        indices = np.random.choice(len(flat), min(10, len(flat)), replace=False)
        
        for i in indices:
            old_val = flat[i]
            flat[i] = old_val + eps
            param[:] = flat.reshape(param.shape)
            loss_plus = cross_entropy_loss(model(x), y)
            flat[i] = old_val
            param[:] = flat.reshape(param.shape)
            grad[i] = (loss_plus - base_loss) / eps
        
        # Update only sampled parameters
        flat[indices] -= lr * grad[indices]
        param[:] = flat.reshape(param.shape)
    
    return base_loss
```

### Putting It All Together

```python
# Training data: a simple repeated pattern
text = "hello world! " * 100
chars = sorted(set(text))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for c, i in char_to_idx.items()}

# Encode
data = np.array([char_to_idx[c] for c in text])

# Create model
model = TinyTransformer(
    vocab_size=len(chars),
    d_model=32,
    n_heads=2,
    d_ff=64,
    n_layers=1,
    max_seq_len=32
)

# Train
seq_len = 16
for step in range(100):
    # Random starting position
    start = np.random.randint(0, len(data) - seq_len - 1)
    x = data[start:start + seq_len]
    y = data[start + 1:start + seq_len + 1]
    
    loss = train_step_numerical(model, x, y, lr=0.1)
    
    if step % 10 == 0:
        print(f"Step {step}, Loss: {loss:.4f}")
```

### Generating Text

```python
def generate(model, start_text, length=50, temperature=1.0):
    """Generate text character by character."""
    tokens = [char_to_idx[c] for c in start_text]
    
    for _ in range(length):
        x = np.array(tokens[-model.max_seq_len:])
        logits = model(x)
        
        # Sample from the last position
        probs = softmax(logits[-1] / temperature)
        next_token = np.random.choice(len(probs), p=probs)
        tokens.append(next_token)
    
    return ''.join(idx_to_char[t] for t in tokens)

# Try it!
print(generate(model, "hel", length=30))
```

## What You Should See

When you run this (and you should run this), the loss will start high—around 2.5 or so, which is roughly $\ln(\text{vocab\_size})$—and gradually decrease. After 100 steps with our crude numerical gradient method, you probably won't see very coherent generations. But with a proper backpropagation implementation or a framework like PyTorch, this exact architecture, trained on a larger dataset for more iterations, will learn meaningful patterns[^3].

## From Toy to GPT

The architecture we've built is, conceptually, the same as GPT. The differences are:

- **Scale**: GPT-3 has 175 billion parameters. We have maybe a few thousand.
- **Data**: GPT trained on hundreds of billions of tokens. We used 1,300 characters.
- **Optimization**: GPT uses carefully tuned Adam with learning rate schedules. We used numerical gradients and hope.
- **Architecture tweaks**: Various normalizations, different activation functions, rotary embeddings instead of absolute positions, etc.

But the core insight—attention is all you need, stacked many times, trained to predict the next token—is exactly what we've implemented. The rest is engineering, compute, and a lot of money.

## Where to Go From Here

If you want to go further:

- **Port to PyTorch**: Replace our manual computations with `torch.nn` modules. The architecture maps directly.
- **Add real training**: Use `loss.backward()` and actual optimizers.
- **Scale up**: Train on real text (Shakespeare, Wikipedia, your own writing) with more layers and dimensions.
- **Read the papers**: "Attention Is All You Need" (2017), the GPT papers, "Language Models are Few-Shot Learners" (GPT-3).

The transformer is, at this point, the foundation of nearly all modern language AI. Understanding it from first principles—as we've done here—means you understand, at some level, the thing that's reshaping how we interact with machines[^4].

Thanks for reading. Go build something.

[^1]: Why characters instead of words? Because characters require no tokenizer, no vocabulary construction, nothing but the raw bytes. It's pedagogically cleaner, even if practically you'd almost always use something like BPE (Byte Pair Encoding) or the GPT tokenizer.

[^2]: This is a terrible way to train neural networks. Each gradient estimate requires N forward passes where N is the number of parameters you're updating. Real implementations use backpropagation, which computes all gradients in a single backward pass using the chain rule. I'm using numerical gradients here only because it requires no additional machinery and makes the training loop transparent.

[^3]: If you want to see this actually work, port the code to PyTorch, replace the numerical gradients with real backprop (`loss.backward(); optimizer.step()`), and train on a few megabytes of text for a few minutes. You'll get a model that generates surprisingly coherent output, even at this tiny scale.

[^4]: Whether this is a good thing or a bad thing or just a thing is a question for another essay. The transformer doesn't care what you think about it. It just predicts the next token, over and over, until something that looks like meaning emerges from the statistical patterns. There's probably a metaphor for consciousness in there somewhere, but I'll leave that to the philosophers, or to the machines that will eventually replace them.
