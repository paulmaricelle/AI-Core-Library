import numpy as np
from .metrics import accuracy, mae, mse, binary_metrics

class Model:
    def __init__(self, sequential):
        self.sequential = sequential


    def train_step(self, X, y, loss):
        y_pred = self.sequential.forward(X)
        loss_val = loss(y_pred, y)

        # Here the gradient is reset at every step, which may
        # not always be the intended purpose
        grad = loss.backward()
        self.sequential.backward(grad)

        return loss_val
        
    def fit(self, X, y, epochs, loss, optimizer, batch_size=1, validation_data = None, patience = 50, accumulation_steps=1, metrics = [], binary_classification_threshold = 0.5, verbose=True):
        optimizer.setup(self.sequential.layers)
        n_samples = X.shape[1]

        #Printing period
        period = max(1, 10**(int(np.log10(epochs)-2)))
        #Early Stopping
        early_stopping = False
        if validation_data != None:
            early_stopping = True
            wait = 0
            best_loss = np.inf

        #Actual training
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[:, indices]
            y_shuffled = y[:, indices]

            loss_value = 0
            self.sequential.set_training(True)

            for i in range(0, n_samples, batch_size):
                if (i // batch_size) % accumulation_steps == 0:
                    optimizer.zero_grad()

                x_batch = X_shuffled[:,  i : i + batch_size]
                y_batch = y_shuffled[:, i : i + batch_size]

                loss_batch = self.train_step(x_batch, y_batch, loss, optimizer)
                actual_batch_size = x_batch.shape[1]
                loss_value += loss_batch * actual_batch_size

                #Handling of accumulation of gradients
                if (i // batch_size + 1) % accumulation_steps == 0:
                    optimizer.step(accumulation_steps)
            
            num_batches = (n_samples + batch_size - 1) // batch_size
            if num_batches % accumulation_steps != 0:
                optimizer.step(num_batches % accumulation_steps)
                optimizer.zero_grad()

            #Early Stopping again
            if early_stopping:
                self.sequential.set_training(False)
                y_pred = self.sequential.forward(validation_data[0])
                validation_loss_value = loss(y_pred, validation_data[1])

                if validation_loss_value < best_loss:
                    best_loss = validation_loss_value
                    wait = 0
                else:
                    wait += 1

                #Metrics on validation
                result = self.compute_metrics(validation_data[0], validation_data[1], metrics, binary_classification_threshold)
                for i in range(len(metrics)):
                    print(metrics[i]+" on validation set is "+result[i])

            #Metrics on training set
            result = self.compute_metrics(X, y, metrics, binary_classification_threshold)
            for i in range(len(metrics)):
                print(metrics[i]+" on training set is "+result[i])
                 
            mean_loss = loss_value / n_samples
            if verbose and epoch % period == 0:
                print(f"Iteration {epoch} completed, loss is {mean_loss}")
                if early_stopping:
                    #There is no need to divide by the number of samples as there is only one batch so it is done instantly
                    print(f"Iteration {epoch} completed, validation loss is {validation_loss_value}")

            if early_stopping and wait >= patience:
                if verbose:
                    print("Early Stopping, patience reached")
                break

    def predict(self, X):
        self.sequential.set_training(False)
        return self.sequential.forward(X)
    
    def compute_metrics(self, X, y, metrics, threshold):
        if len(metrics) > 0:
            y_pred = self.predict(X)
            result = []

            for metric in metrics:
                if metric == "accuracy":
                    result.append(accuracy(y_pred=y_pred, y_true=y))
                if metric == "mae":
                    result.append(mae(y_pred=y_pred, y_true=y))
                if metric == "mse":
                    result.append(mse(y_pred=y_pred, y_true=y))
                if metric == "binary":
                    result.append(binary_metrics(y_pred, y, threshold))
        return result