import numpy as np
from src.ai_lib import layers
from src.ai_lib import Sequential, Model
from src.ai_lib import losses
from src.ai_lib import optimizers
from src.ai_lib import metrics

def load_local_mnist(images_path):
    with open(images_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    
    X = data.reshape(-1, 1, 28, 28) / 255.0
    
    return X.astype(np.float32)

X_train = load_local_mnist('data/mnist/train-images.idx3-ubyte')
X_val = load_local_mnist('data/mnist/t10k-images.idx3-ubyte')

autoencoder = Sequential([
    layers.Conv2d(1, out_channels=8, kernel_size=3, stride=2, padding=1), # Output: (B, 8, 14, 14)
    layers.ReLU(),
    layers.Conv2d(8, 16, kernel_size=3, stride=2, padding=1), # Output: (B, 16, 7, 7)
    layers.ReLU(),
    layers.Flatten(),
    layers.Linear(16 * 7 * 7, 32),

    layers.Linear(32, 16 * 7 * 7, init_method="he"),
    layers.ReLU(),
    layers.Reshape((16, 7, 7)),

    layers.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1), # Output: (B, 8, 14, 14)
    layers.ReLU(),
    layers.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1), # Output: (B, 1, 28, 28)
    layers.Sigmoid()
])

model = Model(autoencoder)
model.fit(X_train, X_train, 10, losses.MSE(), optimizers.Adam(learning_rate=0.0001, weight_decay=0.01), batch_size=16, accumulation_steps=4, validation_data=[X_val, X_val], early_stopping=True, patience=3)

