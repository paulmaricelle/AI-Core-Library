import numpy as np

class Model:
    def __init__(self, sequential):
        self.sequential = sequential


    def train_step(self, X, y, loss, optimizer):
        y_pred = self.sequential.forward(X)
        loss_val = loss(y_pred, y)

        # Here the gradient is reset at every step, which may
        # not always be the intended purpose
        optimizer.zero_grad()
        grad = loss.backward()
        self.sequential.backward(grad)

        optimizer.step()

        return loss_val
        
    def fit(self, X, y, epochs, loss, optimizer, batch_size=1, verbose = True):
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
                x_batch = X_shuffled[:,  i : i + batch_size]
                y_batch = y_shuffled[:, i : i + batch_size]

                loss_batch = self.train_step(x_batch, y_batch, loss, optimizer)
                # If the loss is "non-linear" with respect to the batches,
                # the loss_value will be affected by the batch_size.
                loss_value += loss_batch * batch_size

            if verbose and epoch % period == 0:
                print(f"Iteration {epoch} completed, loss is {loss_value}")               