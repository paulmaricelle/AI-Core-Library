import numpy as np
from .metrics import accuracy, mae, mse, binary_metrics
from .dataLoader import DataLoader

class Model:
    def __init__(self, sequential):
        self.sequential = sequential


    def train_step(self, X, y, loss):
        y_pred = self.sequential.forward(X)
        loss_val = loss(y_pred, y)

        grad = loss.backward()
        self.sequential.backward(grad)

        return loss_val, y_pred
        
    def fit(self, X, y, epochs, loss, optimizer, batch_size=1, validation_data = None, early_stopping=False, patience = 50, accumulation_steps=1, metrics = [], binary_classification_threshold = 0.5, verbose=True):
        optimizer.setup(self.sequential.layers)
        n_samples = X.shape[1]

        #Printing period
        period = max(1, 10**(int(np.log10(epochs)-2)))
        #Early Stopping
        early_stopping = early_stopping and (validation_data != None)
        if early_stopping:
            wait = 0
            best_loss = np.inf

        #Actual training
        for epoch in range(epochs):
            dataLoader = DataLoader(X, y, batch_size=batch_size, shuffle=True)

            loss_value = 0
            self.sequential.set_training(True)
            epoch_metrics = np.zeros((len(metrics),))

            for i, x_batch, y_batch in enumerate(dataLoader):
                if (i // batch_size) % accumulation_steps == 0:
                    optimizer.zero_grad()

                loss_batch, y_pred = self.train_step(x_batch, y_batch, loss)
                actual_batch_size = x_batch.shape[1]
                loss_value += loss_batch * actual_batch_size

                #Handling of accumulation of gradients
                if (i // batch_size + 1) % accumulation_steps == 0:
                    optimizer.step(accumulation_steps)

                for i, metric_fn in enumerate(metrics):
                    batch_metric = metric_fn(y_pred, y_batch, binary_classification_threshold)
                    epoch_metrics[i] += batch_metric * actual_batch_size

            mean_loss = loss_value / n_samples
            epoch_metrics = epoch_metrics / n_samples
            
            num_batches = (n_samples + batch_size - 1) // batch_size
            if num_batches % accumulation_steps != 0:
                optimizer.step(num_batches % accumulation_steps)
                optimizer.zero_grad()

            if validation_data != None:
                validation_loss_value = self.get_validation_loss(validation_data, loss=loss)
            if early_stopping:
                best_loss, wait = self.update_wait(validation_loss_value, best_loss, wait)

            if epoch % period == 0:
                self.log_post_epoch(epoch_metrics, validation_data, mean_loss, metrics, binary_classification_threshold, epoch, verbose, validation_loss_value)

            if early_stopping and wait >= patience:
                if verbose:
                    print("Early Stopping, patience reached")
                break

    def predict(self, X):
        self.sequential.set_training(False)
        return self.sequential.forward(X)
    
    def predict_proba(self, X):
        logits = self.predict(X)
        exp = np.exp(logits - np.max(logits, axis=0, keepdims=True))
        return exp / np.sum(exp, axis=0, keepdims=True)
    
    def compute_metrics(self, X, y, metrics, threshold, batch_size=32):
        dataLoader = DataLoader(X, y, batch_size=batch_size, shuffle=False)
        val_metrics = np.zeros((len(metrics),))
        for X_batch, y_batch in dataLoader:
            y_pred = self.sequential.forward(X_batch)
            actual_batch_size = X_batch.shape[1]

            for j, metric_fn in enumerate(metrics):
                val_metrics[j] += metric_fn(y_pred, y_batch, threshold) * actual_batch_size
        
        return val_metrics / dataLoader.n_samples
    
    def log_post_epoch(self, epoch_metrics, validation_data, mean_loss, metrics, binary_classification_threshold, epoch, verbose, validation_loss_value=None):
        if validation_data != None:
            #Metrics on validation
            val_epoch_metrics = self.compute_metrics(validation_data[0], validation_data[1], metrics, binary_classification_threshold, batch_size=32)
            for i in range(len(metrics)):
                print(f"{metrics[i].__name__} on validation set is {round(val_epoch_metrics[i], 5)}")

        #Metrics on training set
        for i in range(len(metrics)):
            print(f"{metrics[i].__name__} on training set is {round(epoch_metrics[i], 5)}")
                
        if verbose:
            print(f"Iteration {epoch} completed, loss is {mean_loss}")
            if validation_data != None:
                #There is no need to divide by the number of samples as there is only one batch so it is done instantly
                print(f"Iteration {epoch} completed, validation loss is {validation_loss_value}")

    def get_validation_loss(self, validation_data, loss):
        self.sequential.set_training(False)
        y_pred = self.sequential.forward(validation_data[0])
        return loss(y_pred, validation_data[1])
    
    def update_wait(self, validation_loss_value, best_loss, wait):     
        if validation_loss_value < best_loss:
            best_loss = validation_loss_value
            wait = 0
        else:
            wait += 1
        return best_loss, wait