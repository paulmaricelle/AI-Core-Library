import numpy as np

def im2col(X: np.ndarray, kernel_size: int, stride: int, padding: int) -> np.ndarray:
        B, C, H, W = X.shape

        out_H = (H + 2 * padding - kernel_size) // stride + 1
        out_W = (W + 2 * padding - kernel_size) // stride + 1

        if padding > 0:
            X_padded = np.pad(X, ((0,0), (0,0), (padding, padding), (padding, padding)))
        else:
            X_padded = X

        cols = np.zeros((B, C, kernel_size, kernel_size, out_H, out_W))

        # Using for loops here is not much of a loss of time as it is on
        # kernel_size**2 iterations only and it allows for better readability
        # and low RAM usage by simply using views and not actually copying data
        for y in range(kernel_size):
            y_max = y + stride * out_H

            for x in range(kernel_size):
                x_max = x + stride * out_W

                cols[:, :, y, x, :, :] = X_padded[:, :, y:y_max:stride, x:x_max:stride]

        return cols.reshape(B, C * kernel_size * kernel_size, out_H * out_W)

def col2im(cols: np.ndarray, input_shape: int, kernel_size: int, stride: int, padding: int) -> np.ndarray:
    B, C, H, W = input_shape
    out_h = (H + 2 * padding - kernel_size) // stride + 1
    out_w = (W + 2 * padding - kernel_size) // stride + 1

    X_padded = np.zeros((B, C, H + 2 * padding, W + 2 * padding)) 
    cols_reshaped = cols.reshape(B, C, kernel_size, kernel_size, out_h, out_w)

    for y in range(kernel_size):
        y_max = y + stride * out_h
        for x in range(kernel_size):
            x_max = x + stride * out_w 
        
            X_padded[:, :, y:y_max:stride, x:x_max:stride] += cols_reshaped[:, :, y, x, :, :]
    
    if padding > 0:
        return X_padded[:, :, padding:-padding, padding:-padding]
    else:
        return X_padded