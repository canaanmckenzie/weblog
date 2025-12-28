---
title: "Consider the Transformer"
subtitle: "Part 1 of 4 · An introduction, or: why machines learned to pay attention"
description: "An introduction to transformer architecture, or: why machines learned to pay attention."
date: 2025-12-28
template: post
uses_math: true
uses_code: true
---

Here is something that you probably already know but which bears stating anyway, in the manner of all good lectures[^1]: the transformer architecture, which is the thing that makes ChatGPT and Claude and all the other AI assistants possible, is fundamentally a machine that has learned to *pay attention*. And I don't mean this in some loose metaphorical sense, like how we say a thermostat "wants" to keep the room at 72 degrees. I mean that attention—the focusing of computational resources on the parts of an input that matter most for the task at hand—is literally the central mechanism around which the whole thing is built.

The name itself, "transformer," sounds vaguely Optimus Prime-ish, but it comes from the 2017 paper *Attention Is All You Need*[^2], and what the researchers at Google were transforming was the dominant paradigm for processing sequences of things—words, tokens, symbols, whatever. Before transformers, we had recurrent neural networks (RNNs)[^3], which processed sequences the way you or I might read a sentence: one word at a time, left to right, accumulating meaning as we go.

## The Tyranny of Sequential Processing

The problem with RNNs, and this is crucial, is that they suffer from what you might call the tyranny of sequential processing. If you want to understand how the word at position 500 in a document relates to the word at position 3, the network has to iterate through all 497 intervening positions, each time squeezing the accumulated "understanding" through a relatively narrow bottleneck of hidden state. Information gets degraded. Gradients vanish or explode during training. Long-range dependencies—which is just a fancy way of saying "how words far apart in a sentence relate to each other"—become very hard to model.

Consider the sentence: "The cat that the dog that the man owned chased ran away." The grammatical subject of "ran away" is "cat," but there are seven words between them. An RNN trying to connect these distant buddies has to pass information through the entire intervening mess, and by the time it gets there, the signal has often degraded into noise.

## Enter Attention

What the transformer does instead is almost embarrassingly simple once you see it: it allows every position in the sequence to directly attend to every other position. No sequential processing required. No information bottleneck. The word "cat" can directly look at the word "ran" and say, in effect, "Hey, I think we're related."

Mathematically, what this looks like is a weighted sum. For each position in the sequence, we compute how much attention it should pay to every other position, and then we take a weighted combination of all the values based on those attention weights. If this sounds vague, don't worry—Part 2 will get into the grisly details. For now, the intuition is what matters.

The attention mechanism computes something called a **compatibility score** between positions. Positions that are "compatible" (relevant to each other) get high scores; positions that aren't get low scores. These scores become weights, and the weights determine how much each position contributes to the output representation of every other position.

Here's a taste of what the math looks like, just to set the table for later:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Don't panic. $Q$, $K$, and $V$ stand for Query, Key, and Value—three matrices that are computed from the input. The softmax ensures our attention weights sum to 1. The $\sqrt{d_k}$ is a scaling factor that keeps the numbers from getting too big. We'll unpack all of this in the next installment.

## Why This Matters

The shift from RNNs to transformers is not just a technical improvement; it's a conceptual one. RNNs model language as a *process*—a left-to-right traversal through time. Transformers model language as a *structure*—a web of relationships between positions, all computed in parallel.

This parallelism is also why transformers are so well-suited to modern GPUs, which are designed to do many small computations simultaneously rather than a few large ones sequentially. It's one of those happy accidents where a theoretical insight (attention is all you need) happens to align perfectly with the available hardware. Or maybe it's not an accident at all[^4].

In the next part, we'll dive into the self-attention mechanism itself—the engine that drives the whole thing. We'll look at queries, keys, and values; we'll work through the math; and we'll write some actual Python code that implements attention from scratch. Stay tuned.

[^1]: The great irony of lectures is that the things most worth stating explicitly are often the things the lecturer assumes everyone already knows, and the things that get the most elaborate explanations are often the things that could have been left as exercises for the reader. This footnote itself is probably an example of the latter.

[^2]: Vaswani, A., et al. (2017). "Attention Is All You Need." *Advances in Neural Information Processing Systems*. The paper's title is memorable precisely because it sounds like the kind of thing a meditation guru would say, and yet it turns out to be a fairly accurate technical description. The full transformer architecture, while elegant, involves considerably more machinery than just attention—layer normalization, residual connections, positional encodings, feed-forward networks—but attention is indeed the core innovation.

[^3]: And their fancier cousins, LSTMs (Long Short-Term Memory networks), which were specifically designed to address the vanishing gradient problem that plagued vanilla RNNs. LSTMs were, for about a decade, the state of the art for sequence modeling tasks like machine translation and speech recognition. They're ingeniously designed, with gates that control the flow of information through time, and there's something almost poignant about how quickly they were supplanted by the simpler, more parallelizable transformer.

[^4]: There's a whole philosophy of science question lurking here about whether the ideas that succeed are the ones that happen to fit the tools we have, or whether we're genuinely converging on The Right Way To Do Things. The cynical view is that transformers won because GPUs are good at matrix multiplication, not because attention is a fundamental truth about intelligence. The optimistic view is that the universe really is organized around certain principles (parallelism, sparse connectivity, hierarchical abstraction) and we're discovering them. I leave this to the reader's own philosophical inclinations.
