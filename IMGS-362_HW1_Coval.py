'''
Docstring for IMGS-362_HW1_Coval


Fully connected neural network for Fashion MNIST data (clothing instead of numbers)


'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



def accuracy_from_probs(probs, labels): # borrowed helper function from the example
    # Compute accuracy from predicted probabilities and true labels.
    preds = tf.argmax(probs, axis=1, output_type=tf.int32)
    labels = tf.cast(labels, tf.int32)
    correct = tf.equal(preds, labels)
    return tf.reduce_mean(tf.cast(correct, tf.float32))




def main():

    # -----------------load and process fashion MNIST -----------------------

    (x_train_full,y_train_full), (x_test,y_test) = tf.keras.datasets.fashion_mnist.load_data()

    print(f"test set shape: {x_test.shape}, {y_test.shape}")

    # normalize images to 0-1
    x_train_full = x_train_full.astype(np.float32)/255.0
    x_test = x_test.astype(np.float32)/255.0

    # split data for training and validation
    x_train, x_val = x_train_full[:50000], x_train_full[50000:]
    y_train,y_val = y_train_full[:50000], y_train_full[50000:]

    print("Training set:", x_train.shape, y_train.shape)
    print("Validation set:", x_val.shape, y_val.shape)
    print("Test set:", x_test.shape, y_test.shape)

    # flatten images into 1-D vectors

    x_train_flat = x_train.reshape(-1, 28 * 28)
    x_val_flat = x_val.reshape(-1, 28 * 28)
    x_test_flat = x_test.reshape(-1, 28 * 28)

    print("Flattened training set shape:", x_train_flat.shape)

    # use built-in datasets from tensorflow for best efficiency

    batch_size = 128   # <<----------hyperparameter 

    train_ds = tf.data.Dataset.from_tensor_slices((x_train_flat, y_train)).shuffle(10000).batch(batch_size) # shuffle to randomize order
    val_ds   = tf.data.Dataset.from_tensor_slices((x_val_flat, y_val)).batch(batch_size)
    test_ds  = tf.data.Dataset.from_tensor_slices((x_test_flat, y_test)).batch(batch_size)


    # ---------------------- define fully connected network---------------------





if __name__ == "__main__":
    main()
    