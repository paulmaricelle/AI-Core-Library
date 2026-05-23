# AI-Core-Library

**AI-Core-Library** est un framework de Deep Learning complet développé entièrement de zéro (from scratch) en utilisant exclusivement **Python et NumPy**. 

L'objectif de ce projet est pédagogique et technique : démystifier la "boîte noire" des frameworks industriels (comme PyTorch ou TensorFlow) en réimplémentant manuellement le graphe de calcul, la rétropropagation des gradients, les calculs tensoriels optimisés et les architectures avancées (de la Vision par Ordinateur jusqu'aux Modèles de Langage LLMs modernes), ainsi que découvrir par moi-même de nouvelles architectures et algorithmes.

## Idées Principales

- **Architecture Orientée Objet :** Création de modèles intuitifs via des classes `Sequential` et `Model`, facilitant l'assemblage de réseaux complexes.
- **Rétropropagation Manuelle :** Calcul exact des gradients pour chaque couche, fonction d'activation, et mécanisme d'attention.
- **Modèles de Langage (LLMs) :** Implémentation d'architectures état de l'art (type Llama/Mistral) optimisées pour l'entraînement autoregressif de la génération de texte (ex: dataset TinyStories).
- **Mixture of Experts (MoE) :** Support natif du routage dynamique (Top-2) avec tri des tokens (Token Sorting) optimisé pour CPU, incluant la Loss Auxiliaire (Load Balancing) pour éviter l'effondrement du routeur.
- **Convolutions Vectorisées :** Implémentation industrielle des opérations spatiales via `im2col` et `col2im` pour des performances optimales sans boucles `for` naïves.
- **Modèles Génératifs :** Support natif de la Reparamétrisation (KL-Divergence) pour la création d'Autoencodeurs Variationnels (VAE).
- **Planification de l'Apprentissage :** Intégration de LR Schedulers (Cosine Decay & Linear Warmup) injectés dynamiquement dans la boucle d'entraînement.
- **Persistance et Checkpointing :** Sauvegarde et chargement des poids du modèle, ainsi que de l'état complet du système (Optimiseur, Steps) pour sécuriser les longs entraînements.
- **Gestion de la Mémoire :** `DataLoader` optimisé par générateurs pour éviter la saturation de la RAM lors du traitement des batchs.

---

## Modules Implémentés

### Couches (Layers)
- **Transformers & LLMs (Nouveau) :**
  - `Embedding` et `TiedOutputProjection` (Weight Tying).
  - `MultiSelfAttentionHead` avec support du **Grouped-Query Attention (GQA)**.
  - `PositionalEncoding` incluant **RoPE** (Rotary Positional Embeddings).
  - `SwiGLU` (Feed-Forward network optimisé).
  - `RMSNorm` (Root Mean Square Normalization).
  - `TransformerBlock` et `MoEBlock` (Mixture of Experts agnostic builder).
- **Classiques :** `Linear` (Dense), `Flatten`, `Reshape`, `Dropout`.
- **Convolutives :** `Conv2d`, `ConvTranspose2d`, `MaxPooling2D`.
- **Avancées :** `ResidualBlock` (Skip connections), `LayerNormalization`, `SamplingLayer` (Espace latent VAE).
- **Activations :** `ReLU`, `Sigmoid`, `Tanh`.

### Fonctions de Perte (Losses)
- `MSE` (Mean Squared Error)
- `BinaryCrossEntropy` (BCE)
- `SoftmaxCrossEntropy` (Classification multiclasse et modélisation de langage)
- *Gestion dynamique des Auxiliary Losses (ex: Load Balancing pour MoE).*

### Optimiseurs
- `SGD` (Stochastic Gradient Descent)
- `Momentum`
- `Adam` (Avec gestion complète du *Weight Decay* et des moments d'ordre 1 et 2)

### Planificateurs (LR Schedulers)
- `LRScheduler` (Classe abstraite de base)
- `CosineWarmupScheduler` (Montée en régime linéaire suivie d'une décroissance en cosinus, préférable pour la convergence des LLMs)

### Outils & Prétraitement
- Implémentations d'inférence autoregressive avec gestion de la température (`generate`).
- Tokenizers et DataLoaders adaptables pour la vision et le texte.