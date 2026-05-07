# AI-Core-Library

**AI-Core-Library** est un framework de Deep Learning complet développé entièrement de zéro (from scratch) en utilisant exclusivement **Python et NumPy**. 

L'objectif de ce projet est pédagogique et technique : démystifier la "boîte noire" des frameworks industriels (comme PyTorch ou TensorFlow) en réimplémentant manuellement le graphe de calcul, la rétropropagation des gradients , les calculs tensoriels optimisés et les architectures avancées, ainsi que découvrir pour moi des nouvelles architectures et algorithmes.

## Idées Principales

- **Architecture Orientée Objet :** Création de modèles intuitifs via une classe `Sequential` et `Model`.
- **Rétropropagation Manuelle :** Calcul exact des gradients pour chaque couche et fonction d'activation.
- **Convolutions Vectorisées :** Implémentation industrielle des opérations spatiales via `im2col` et `col2im` pour des performances optimales sans boucles `for` naïves.
- **Modèles Génératifs :** Support natif de la Reparamétrisation (KL-Divergence) pour la création d'Autoencodeurs Variationnels (VAE).
- **Persistance des Données :** Sauvegarde et chargement des poids du modèle au format compressé `.npz` avec gestion hiérarchique des noms de paramètres.
- **Gestion de la Mémoire :** `DataLoader` optimisé par générateurs pour éviter la saturation de la RAM lors du traitement des batchs.

---

## Modules Implémentés

### Couches
- **Classiques :** `Linear` (Dense), `Flatten`, `Reshape`, `Dropout`.
- **Convolutives :** `Conv2d`, `ConvTranspose2d`, `MaxPooling2D`.
- **Avancées :** `ResidualBlock` (Skip connections), `LayerNormalization`, `SamplingLayer` (Espace latent VAE).
- **Activations :** `ReLU`, `Sigmoid`, `Tanh`.

### Fonctions de Perte (Losses)
- `MSE` (Mean Squared Error)
- `BinaryCrossEntropy` (BCE)
- `SoftmaxCrossEntropy` (Classification multiclasse optimisée et stable)

### Optimiseurs
- `SGD` (Stochastic Gradient Descent)
- `Momentum`
- `Adam` (Avec gestion du *Weight Decay*)
