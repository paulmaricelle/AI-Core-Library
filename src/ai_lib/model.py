import numpy as np

class Model:
    def __init__(self, sequential):
        self.sequential = sequential


    def train_step(self, X, y, loss, optimizer):
        y_pred = self.sequential.forward(X)
        loss_val = loss(y_pred, y)

        # Here the gradient is reset at every step, which may
        # not always be the intended purpose
        grad = loss.backward()
        self.sequential.backward(grad)

        return loss_val
        
    def fit(self, X, y, epochs, loss, optimizer, batch_size=1, accumulation_steps=1, verbose=True):
        optimizer.setup(self.sequential.layers)
        n_samples = X.shape[1]

        #Printing period
        period = max(1, 10**(int(np.log10(epochs)-2)))

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[:, indices]
            y_shuffled = y[:, indices]

            loss_value = 0
            for i in range(0, n_samples, batch_size):
                if (i // batch_size) % accumulation_steps == 0:
                    optimizer.zero_grad()

                x_batch = X_shuffled[:,  i : i + batch_size]
                y_batch = y_shuffled[:, i : i + batch_size]

                loss_batch = self.train_step(x_batch, y_batch, loss, optimizer)
                actual_batch_size = x_batch.shape[1]
                loss_value += loss_batch * actual_batch_size

            if (i // batch_size + 1) % accumulation_steps == 0:
                optimizer.step(accumulation_steps)

            if verbose and epoch % period == 0:
                mean_loss = loss_value / n_samples
                print(f"Iteration {epoch} completed, loss is {mean_loss}")               