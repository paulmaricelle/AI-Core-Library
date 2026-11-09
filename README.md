# AI-Core-Library

**A deep-learning framework built from first principles in pure NumPy: no autograd, no PyTorch.** Every forward and backward pass, from convolutions to grouped-query attention, is derived and implemented by hand. The library goes from MLPs, CNNs and VAEs all the way to a modern LLM stack: RoPE, GQA, RMSNorm, SwiGLU, KV caching with attention sinks, and a Mixtral-style sparse Mixture-of-Experts layer. It trains decoder-only language models on TinyStories end to end, from BPE tokenization to text generation: a 4.8M-parameter dense model and a 6.7M-parameter sparse MoE model with only 2.0M parameters active per token. Built to understand, line by line, everything that modern frameworks abstract away.

## Highlights

- **Hand-derived backpropagation everywhere.** Each layer implements its own `forward` and `backward`, including attention with RoPE, grouped-query attention, SwiGLU, RMSNorm, im2col convolutions, transposed convolutions and the VAE reparameterisation trick.
- **A modern LLM stack.** Llama/Mistral-style pre-norm Transformer blocks, rotary embeddings, grouped-query attention (8 query heads sharing 4 KV heads), SwiGLU feed-forward, RMSNorm, tied input/output embeddings.
- **Efficient inference.** KV cache for incremental decoding, sliding-window attention with attention-sink tokens, top-k and temperature sampling.
- **Mixture of Experts.** Mixtral-style sparse layer: learned router, top-2 routing over 8 SwiGLU experts, Switch-style load-balancing term applied to the router gradient, hand-written backward through the routing. Used in every block of a trained 6-layer MoE language model.
- **A complete training toolkit.** Adam with decoupled weight decay, cosine schedule with linear warmup, gradient accumulation, early stopping, checkpointing to `.npz`, random-window sequence data loading.

## Features

- **Core**: `Layer` interface with manual `forward`/`backward`, `Sequential`, `Model` (`fit`, gradient accumulation, early stopping, validation, `save_weights`/`load_weights`).
- **Layers**: Linear, Conv2d (im2col/col2im), ConvTranspose2d, MaxPooling2D, Flatten, Reshape, Dropout, LayerNorm, RMSNorm, ResidualBlock, ReLU/Sigmoid/Tanh, SamplingLayer (VAE reparameterisation + KL).
- **Transformer**: Embedding, tied output projection, sinusoidal encoding, multi-head attention, RoPE attention, GQA + RoPE attention, SwiGLU, pre-norm `TransformerBlock` built from pluggable attention / norm / FFN builders.
- **Inference**: KV cache; sliding window with 4 attention-sink tokens; top-k / temperature sampling.
- **Mixture of Experts**: `MoEBlock` (router, top-k routing, load-balancing gradient, hand-written backward).
- **Losses**: MSE, Binary cross-entropy, Softmax cross-entropy.
- **Optimisers and schedules**: SGD, Momentum, Adam (weight decay); cosine decay with linear warmup.
- **Data and tokenisation**: `DataLoader`, `SequenceDataLoader`; character-level tokenizer; BPE through HuggingFace `tokenizers` / `tiktoken` (the tokenizer is the only component not written from scratch).

## Results

Numbers from the saved notebook outputs (single runs). Parameter counts are computed from each model's configuration.

| Model | Architecture | Data | Params | Val. loss (nats/token) | Val. perplexity | Val. next-token accuracy |
|---|---|---|---|---|---|---|
| **microgpt_v3** | 6 blocks, d=256, GQA (8 heads / 4 KV), RoPE, RMSNorm, SwiGLU, tied embeddings, BPE vocabulary of 2,000 | TinyStories (50k stories) | **4.84M** | **2.61** | **13.6** | **43%** |
| **microgpt_v3_moe** | 6 blocks, d=128, GQA (8 heads / 4 KV), RoPE, RMSNorm, **MoE in every block: 8 SwiGLU experts, top-2 routing**, tied embeddings, BPE vocabulary of 1,000 | TinyStories (50k stories) | **6.72M total / 2.00M active per token** | 2.26 | 9.6 | 48% |
| Convolutional VAE | 2-d latent, conv encoder / transposed-conv decoder | MNIST | 15.6k | 151.2 (per image) | — | — |

The dense and MoE models use different vocabularies (2,000 vs. 1,000 BPE tokens) and widths, so their losses are not directly comparable; the MoE run is a short training run (200 optimisation steps), reported at its best validation epoch.

**Earlier milestones.** `microgpt_v1` (character-level GPT, ~0.1M parameters) and `microgpt_v2` (BPE + RoPE, ~1.2M parameters) were trained on TinyShakespeare to validate the stack step by step. Their losses are not comparable with v3: they use different vocabularies (65 characters, 1,500 and 2,000 BPE tokens respectively), so their losses and perplexities are measured over different units. v2 overfitted the 1.1M-character corpus, which motivated the move to TinyStories and a larger, Llama-style model in v3.

## Layout

```
src/ai_lib/
  model.py            training loop, checkpointing
  layers/             all layers (attention, moe, swiglu, normalization, conv2d, ...)
  losses/  optimizers/  lrSchedulers.py  dataLoader.py  preprocessing/
microgpt_v3.ipynb     GQA + RoPE + RMSNorm + SwiGLU language model, TinyStories
microgpt_v3_moe.ipynb Mixtral-style sparse MoE language model (8 experts, top-2), TinyStories
microgpt_v2.ipynb     BPE + RoPE, TinyShakespeare (earlier API)
microgpt_v1.ipynb     character-level GPT, TinyShakespeare (earlier API)
mnist_vae.ipynb       convolutional VAE on MNIST
mnist_*.py            MLP, CNN and autoencoder examples on MNIST
```

## Getting started

```bash
git clone https://github.com/paulmaricelle/AI-Core-Library.git
cd AI-Core-Library
pip install -r requirements.txt
jupyter notebook microgpt_v3.ipynb
```

Run everything from the repository root (imports use `src.ai_lib`).

**Data** (not committed, see `.gitignore`):
- *TinyStories*: `microgpt_v3.ipynb` contains a (commented-out) cell that downloads the first 50k stories with HuggingFace `datasets` and writes `data/tinystories_extract.txt`; uncomment it on first run.
- *TinyShakespeare*: downloaded automatically by `microgpt_v1.ipynb` / `microgpt_v2.ipynb`.
- *MNIST*: place the four original `idx` files under `data/mnist/`.

## Roadmap

- Finite-difference gradient checks for every layer (attention, SwiGLU, MoE router included).
- Longer MoE training, expert-utilisation logging, and a dense-vs-MoE comparison at equal active parameters and equal vocabulary.
- Port v1/v2 notebooks to the current builder-based `TransformerBlock` API.

## License

MIT
