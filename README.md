# AI-Core-Library

**A deep-learning framework built from first principles in pure NumPy: no autograd, no PyTorch.** Every forward and backward pass, from convolutions to grouped-query attention, is derived and implemented by hand. The library goes from MLPs, CNNs and VAEs all the way to a modern LLM stack: RoPE, GQA, RMSNorm, SwiGLU, KV caching with attention sinks, and a Mixtral-style sparse Mixture-of-Experts layer. It trains a 4.8M-parameter decoder-only language model on TinyStories end to end, from BPE tokenization to text generation. Built to understand, line by line, everything that modern frameworks abstract away.

## Highlights

- **Hand-derived backpropagation everywhere.** Each layer implements its own `forward` and `backward`, including attention with RoPE, grouped-query attention, SwiGLU, RMSNorm, im2col convolutions, transposed convolutions and the VAE reparameterisation trick.
- **A modern LLM stack.** Llama/Mistral-style pre-norm Transformer blocks, rotary embeddings, grouped-query attention (8 query heads sharing 4 KV heads), SwiGLU feed-forward, RMSNorm, tied input/output embeddings.
- **Efficient inference.** KV cache for incremental decoding, sliding-window attention with attention-sink tokens, top-k and temperature sampling.
- **Mixture of Experts.** Mixtral-style sparse layer: learned router, top-2 routing over 8 SwiGLU experts, load-balancing auxiliary loss, hand-written backward through the router.
- **A complete training toolkit.** Adam with decoupled weight decay, cosine schedule with linear warmup, gradient accumulation, early stopping, checkpointing to `.npz`, random-window sequence data loading.

## Features

- **Core**: `Layer` interface with manual `forward`/`backward`, `Sequential`, `Model` (`fit`, gradient accumulation, early stopping, validation, `save_weights`/`load_weights`).
- **Layers**: Linear, Conv2d (im2col/col2im), ConvTranspose2d, MaxPooling2D, Flatten, Reshape, Dropout, LayerNorm, RMSNorm, ResidualBlock, ReLU/Sigmoid/Tanh, SamplingLayer (VAE reparameterisation + KL).
- **Transformer**: Embedding, tied output projection, sinusoidal encoding, multi-head attention, RoPE attention, GQA + RoPE attention, SwiGLU, pre-norm `TransformerBlock` built from pluggable attention / norm / FFN builders.
- **Inference**: KV cache; sliding window with 4 attention-sink tokens; top-k / temperature sampling.
- **Mixture of Experts**: `MoEBlock` (router, top-k routing, load-balancing loss, hand-written backward).
- **Losses**: MSE, Binary cross-entropy, Softmax cross-entropy.
- **Optimisers and schedules**: SGD, Momentum, Adam (weight decay); cosine decay with linear warmup.
- **Data and tokenisation**: `DataLoader`, `SequenceDataLoader`; character-level tokenizer; BPE through HuggingFace `tokenizers` / `tiktoken` (the tokenizer is the only component not written from scratch).

## Results

Numbers from the saved notebook outputs (single runs). Parameter counts are computed from each model's configuration.

| Model | Architecture | Data | Params | Val. loss (nats/token) | Val. perplexity | Val. accuracy (next token) |
|---|---|---|---|---|---|---|
| **microgpt_v3** | 6 blocks, d=256, GQA (8 heads / 4 KV), RoPE, RMSNorm, SwiGLU, tied embeddings, BPE 2k | TinyStories (50k stories) | **4.84M** | **2.61** | **13.6** | 43% |
| microgpt_v2 | 4 blocks, d=128, RoPE, BPE 1.5k | TinyShakespeare | ~1.2M | 4.22 | 68 | 25% |
| microgpt_v1 | 2 blocks, d=64, sinusoidal positions, characters | TinyShakespeare | ~0.1M | 1.71 | 5.5 | 50% |
| Convolutional VAE | 2-d latent, conv encoder / transposed-conv decoder | MNIST | 15.6k | 151.2 (per image) | — | — |

v1 and v2 are earlier milestones; v2 overfits the 1.1M-character corpus (train loss 2.88 vs. val. 4.22), which motivated the move to TinyStories and a larger model in v3. Perplexities are not comparable across tokenizers.

## Layout

```
src/ai_lib/
  model.py            training loop, checkpointing
  layers/             all layers (attention, moe, swiglu, normalization, conv2d, ...)
  losses/  optimizers/  lrSchedulers.py  dataLoader.py  preprocessing/
microgpt_v3.ipynb     GQA + RoPE + RMSNorm + SwiGLU language model, TinyStories
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
- Train a Mixture-of-Experts language model and compare it with the dense v3 at equal active parameters.
- Port v1/v2 notebooks to the current builder-based `TransformerBlock` API.

## License

MIT
