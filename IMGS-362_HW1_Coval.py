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
    # Architecture: 784 --> 200 --> 200 --> 10

    fc_model = tf.keras.Sequential(
    [
        # Input: vectors of length 784
        tf.keras.layers.Input(shape=(784,), name="input_flattened"),

        # First hidden dense layer with 200 units and ReLU activation
        tf.keras.layers.Dense(200, activation="relu", name="dense_1"),  # decrease hidden layers size for computational speed

        # Second hidden dense layer with 200 units and ReLU activation
        tf.keras.layers.Dense(200, activation="relu", name="dense_2"),

        # Output layer with 10 units (one per digit class) and Softmax
        # Softmax converts logits to a probability distribution over classes.
        tf.keras.layers.Dense(10, activation="softmax", name="output_softmax"),
    ],
    name="FullyConnected",
   )

    fc_model.summary()

    # Loss function: SparseCategoricalCrossentropy (integer labels 0-9)
    loss_fn_fc = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    # Optimizer: Adam (possibly experiment with different optimizers)
    optimizer_fc = tf.keras.optimizers.Adam(learning_rate=1e-5)    # tuned 100x slower from example        #<--------hyperparameter


    # initialize lists for metrics
    fc_train_loss_history = []
    fc_val_loss_history = []
    fc_train_acc_history = []
    fc_val_acc_history = []


    # number of epochs of training
    epochs = 10   


    # ------------------------ training and validation------------------------
    for epoch in range(1, epochs + 1):
        print(f"\n=== [FC Network] Epoch {epoch}/{epochs} ===")

        # ---------- Training phase ----------
        train_losses = []
        train_accuracies = []

        for step, (x_batch, y_batch) in enumerate(train_ds):
            with tf.GradientTape() as tape:
                # Forward pass
                probs = fc_model(x_batch, training=True)
                # Compute loss
                loss_value = loss_fn_fc(y_batch, probs)

            # Compute gradients and update weights
            grads = tape.gradient(loss_value, fc_model.trainable_variables)
            optimizer_fc.apply_gradients(zip(grads, fc_model.trainable_variables))

            # Compute accuracy
            acc = accuracy_from_probs(probs, y_batch)

            train_losses.append(loss_value.numpy())
            train_accuracies.append(acc.numpy())

            if step % 100 == 0:
                print(f"  Step {step:03d} - Batch loss: {loss_value:.4f}, accuracy: {acc:.4f}")   

        # Aggregate training metrics
        epoch_train_loss = np.mean(train_losses)
        epoch_train_acc = np.mean(train_accuracies)


        # ---------- Validation phase ----------
        val_losses = []
        val_accuracies = []

        for x_batch_val, y_batch_val in val_ds:
            probs_val = fc_model(x_batch_val, training=False)
            val_loss_value = loss_fn_fc(y_batch_val, probs_val)
            val_acc = accuracy_from_probs(probs_val, y_batch_val)

            val_losses.append(val_loss_value.numpy())
            val_accuracies.append(val_acc.numpy())

        epoch_val_loss = np.mean(val_losses)
        epoch_val_acc = np.mean(val_accuracies)

        # Store history
        fc_train_loss_history.append(epoch_train_loss)
        fc_val_loss_history.append(epoch_val_loss)
        fc_train_acc_history.append(epoch_train_acc)
        fc_val_acc_history.append(epoch_val_acc)

        print(
            f"Epoch {epoch}: "
            f"Train loss = {epoch_train_loss:.4f}, Train acc = {epoch_train_acc:.4f} | "
            f"Val loss = {epoch_val_loss:.4f}, Val acc = {epoch_val_acc:.4f}"
        )
        # ---- end of epoch loop ----

    # --------- plot results ---------------------------------------

    epochs_range_fc = range(1, epochs + 1)

    plt.figure()
    plt.plot(epochs_range_fc, fc_train_loss_history, label="Train Loss")
    plt.plot(epochs_range_fc, fc_val_loss_history, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Fully Connected Network - Loss per Epoch")
    plt.legend()
    plt.grid(True)
    plt.show()
    

    plt.figure()
    plt.plot(epochs_range_fc, fc_train_acc_history, label="Train Accuracy")
    plt.plot(epochs_range_fc, fc_val_acc_history, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Fully Connected Network - Accuracy per Epoch\n2 inner layers of 200 neurons\nLearning rate: 1e-5")
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.savefig("2 layers 200 neurons 1e-5 learn rate 10 epochs.png")




if __name__ == "__main__":
    main()
    