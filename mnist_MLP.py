import numpy as np
from src.ai_lib import layers
from src.ai_lib import Sequential, Model
from src.ai_lib import losses
from src.ai_lib import optimizers
from src.ai_lib import metrics

def load_local_mnist(images_path, labels_path):
    with open(images_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    
    X = data.reshape(-1, 784) / 255.0

    with open(labels_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)

    Y = np.eye(10)[labels]
    
    return X.astype(np.float32), Y.astype(np.float32)

X_train, Y_train = load_local_mnist('data/mnist/train-images.idx3-ubyte', 'data/mnist/train-labels.idx1-ubyte')
X_val, Y_val = load_local_mnist('data/mnist/t10k-images.idx3-ubyte', 'data/mnist/t10k-labels.idx1-ubyte')

network = Sequential([layers.Linear(784, 256, "xavier"), layers.LayerNormalization(256),
                      layers.ReLU(), layers.Dropout(0.4), layers.Linear(256, 128),
                      layers.LayerNormalization(128), layers.ReLU(), layers.Dropout(0.4),
                      layers.Linear(128, 10)])

model = Model(network)
#This achieves 98% accuracy on training set
try:
    model.fit(X_train, Y_train, 50, losses.SoftmaxCrossEntropy(), optimizers.Adam(learning_rate=0.0001, weight_decay=0.01), batch_size=32, accumulation_steps=4, validation_data=[X_val, Y_val], early_stopping=True, patience=20, metrics=[metrics.accuracy])
except KeyboardInterrupt:
    print("Arrêt manuel détecté.")
except Exception as e:
    print(f"L'erreur réelle était : {e}")

