---
title: "Attention, Please"
subtitle: "Part 2 of 4 · The mathematics of giving a damn"
description: "A deep dive into the self-attention mechanism, with math and Python code."
date: 2025-12-28
template: post
uses_math: true
uses_code: true
---

Okay. So. Attention. In the previous installment we talked around the thing, gestured at its shape, promised that the math would come later. Well, later is now[^1].

The core idea of self-attention is this: given a sequence of vectors (representing words or tokens or whatever), we want to compute a new sequence of vectors where each position is a weighted combination of *all* positions in the input. The weights—how much each position contributes—are determined by a learned compatibility function between positions.

## The Three Projections: Query, Key, Value

Let's say we have an input sequence $X$ of shape $(n, d)$—that's $n$ positions, each represented by a $d$-dimensional vector[^2]. The first thing we do is project $X$ into three different spaces:

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

Here $W_Q$, $W_K$, and $W_V$ are learned weight matrices, each of shape $(d, d_k)$ where $d_k$ is the dimension of our key/query space (often $d_k = d / h$ where $h$ is the number of attention heads, but we'll get to that). The names come from a database analogy:

- **Query**: "What am I looking for?"
- **Key**: "What do I have to offer?"
- **Value**: "If you're interested, here's what I'll give you."

Each position broadcasts a Query ("I'm looking for something") and a Key ("Here's my identifier"), and the compatibility between a Query at position $i$ and a Key at position $j$ determines how much of position $j$'s Value gets mixed into position $i$'s output.

## The Compatibility Score

How do we measure compatibility? The transformer uses the simplest possible thing: a dot product. The Query at position $i$ is a vector; the Key at position $j$ is a vector; we take their dot product. High dot product means "these vectors point in similar directions," which we interpret as "these positions are relevant to each other."

But here's the thing: dot products can get very large when the vectors are high-dimensional. If $Q$ and $K$ both have entries drawn from a distribution with variance 1, then the dot product $Q \cdot K$ has variance proportional to $d_k$. Large values going into a softmax get pushed toward 0 or 1, which makes gradients very small (this is called saturation). So we scale:

$$\text{score}(Q, K) = \frac{QK^T}{\sqrt{d_k}}$$

That $\sqrt{d_k}$ keeps the variance of the scores approximately constant regardless of the dimension. It's a small thing, but the kind of small thing that determines whether training actually converges[^3].

## From Scores to Weights: Softmax

We now have an $n \times n$ matrix of compatibility scores. Each row represents one position's Query; each column represents one position's Key. The score at row $i$, column $j$ is how much position $i$'s Query likes position $j$'s Key.

But scores aren't weights. Weights need to sum to 1 (so we get a proper weighted average) and be non-negative (so we don't accidentally subtract information). The softmax function gives us exactly this:

$$\text{Attention Weights} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$$

The softmax is applied row-wise: for each Query position, we get a probability distribution over all Key positions. These probabilities tell us how to weight the Values.

## Computing the Output

Now we just multiply the attention weights (an $n \times n$ matrix) by the Values (an $n \times d_v$ matrix):

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Each row of the output is a weighted combination of all the Value vectors, where the weights come from the softmax of compatibility scores. That's it. That's the core of self-attention.

## Let's Build It in Python

Enough abstraction. Here's a minimal implementation of scaled dot-product attention in NumPy:

```python
import numpy as np

def softmax(x, axis=-1):
    """Numerically stable softmax."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    """
    Compute scaled dot-product attention.
    
    Args:
        Q: Queries, shape (n, d_k)
        K: Keys, shape (n, d_k)  
        V: Values, shape (n, d_v)
    
    Returns:
        Output, shape (n, d_v)
        Attention weights, shape (n, n)
    """
    d_k = Q.shape[-1]
    
    # Compute compatibility scores
    scores = Q @ K.T / np.sqrt(d_k)
    
    # Convert to probability distribution
    attention_weights = softmax(scores)
    
    # Weighted combination of values
    output = attention_weights @ V
    
    return output, attention_weights
```

Let's test it with a tiny example—a 4-word sentence where each word is represented by a 3-dimensional vector:

```python
# Fake embeddings for: ["The", "cat", "sat", "down"]
X = np.array([
    [1.0, 0.0, 0.0],   # The
    [0.0, 1.0, 0.2],   # cat  
    [0.1, 0.2, 1.0],   # sat
    [0.0, 0.0, 1.0],   # down
])

# In practice these are learned; here we just use identity
W_Q = W_K = W_V = np.eye(3)

Q = X @ W_Q
K = X @ W_K
V = X @ W_V

output, weights = scaled_dot_product_attention(Q, K, V)

print("Attention weights:")
print(np.round(weights, 2))
# Each row shows how much each position attends to others
```

When you run this, you'll see that each row of the attention weights sums to 1, and positions with similar embeddings (like "sat" and "down," which both have high values in the third dimension) will attend more strongly to each other[^4].

## Multi-Head Attention: Why One Is Not Enough

Real transformers don't use just one attention computation—they use several in parallel, called "heads." Each head has its own $W_Q$, $W_K$, $W_V$ matrices. The intuition is that different heads can learn to attend to different things: one head might focus on syntactic relationships, another on semantic similarity, another on positional patterns.

The outputs of all heads are concatenated and then projected back down to the model dimension:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W_O$$

where each $\text{head}_i = \text{Attention}(XW_Q^i, XW_K^i, XW_V^i)$.

We'll implement this fully in Part 3 when we build the complete transformer architecture. For now, just know that more heads generally means more expressive power, up to a point[^5].

## What Have We Learned?

Self-attention is a mechanism that:

- Projects inputs into Query, Key, and Value representations
- Computes compatibility between positions via scaled dot products
- Uses softmax to convert scores to weights
- Produces outputs that are weighted combinations of Values

In the next part, we'll zoom out and see where attention fits into the larger transformer architecture: positional encodings, layer normalization, residual connections, and the feed-forward networks that sit between attention layers. Stay attentive.

[^1]: There's a pedagogical question about whether to show the math first and then build intuition, or to build intuition first and then show the math. I have chosen the latter, mostly because that's how I actually learned it, but partly because starting with equations tends to trigger a certain kind of survival-mode skimming that is not conducive to actual understanding. If you are a math-first person, I apologize for making you wait.

[^2]: In practice, these vectors are embeddings—learned representations where semantically similar words end up close together in the high-dimensional space. The dimension $d$ is usually something like 512 or 768 or 1024. GPT-3 uses 12,288, which is absurd but apparently necessary when you have 175 billion parameters to fill.

[^3]: The original paper describes this scaling as "preventing the dot products from growing large in magnitude, which would push the softmax function into regions where it has extremely small gradients." This is the kind of sentence that makes perfect sense once you know what all the words mean and no sense at all before that. Gradient-based training is very sensitive to the scale of numbers flowing through the network. Too big and things explode; too small and things vanish. The whole art of modern deep learning is keeping everything in a Goldilocks zone.

[^4]: This is a somewhat circular demonstration since we're using the input embeddings directly as Q, K, V (with identity projection matrices). In a real network, the learned projections would create much richer and more interesting attention patterns. But you have to start somewhere.

[^5]: The original transformer used 8 heads. BERT uses 12. GPT-3 uses 96. There's no theoretical answer for how many heads are optimal—it depends on the task, the model size, the data, and probably the phase of the moon. The general principle is that more heads allow more diverse attention patterns, but at some point you hit diminishing returns. Also, the total dimension (d_k × h) is typically kept constant, so more heads means smaller per-head dimensions, which has its own tradeoffs.
