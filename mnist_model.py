import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from MnistDataloader import MnistDataloader
from NeuralNetClasses import Layer_Dense, Activation_ReLU, Activation_Softmax, Loss_CategoricalCrossEntropy, TestMNIST

os.environ["OMP_NUM_THREADS"] = "8"

training_images_filepath = 'data/train-images-idx3-ubyte/train-images-idx3-ubyte'
training_labels_filepath = 'data/train-labels-idx1-ubyte/train-labels-idx1-ubyte'
test_images_filepath = 'data/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte'
test_labels_filepath = 'data/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'

mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

# Formatting data
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])

one_hot = np.zeros((60000,10))
one_hot[np.arange(60000), y_train] = 1
y_train = one_hot

# Initializing layers and activation functions
dense1 = Layer_Dense(784, 128)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(128, 64)
activation2 = Activation_ReLU()
dense3 = Layer_Dense(64, 10)
activation3 = Activation_Softmax()

loss_function = Loss_CategoricalCrossEntropy()

epochs = 1000
alpha = 0.000000001

for epoch in range(epochs):
    # Forward pass
    dense1.forward(x_train)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)

    loss = loss_function.calculate(activation3.output, y_train)
    
    # Backward pass
    loss_function.backward(activation3.output, y_train)
    dense3.backward(loss_function.dinputs, alpha)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs, alpha)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs, alpha)

    predictions = np.argmax(activation3.output, axis=1)
    y_true = np.argmax(y_train, axis=1)
    print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {np.mean(predictions == y_true):.4f}")


model = ((dense1.weights, dense1.biases), (dense2.weights, dense2.biases), (dense3.weights, dense3.biases))
n_inputs = (784, 128, 64)
n_neurons = (128, 64, 10)
s = x_test.shape
x_test = x_test.reshape(s[0], s[1] * s[2])

tester = TestMNIST(model, n_inputs, n_neurons)
accuracy = tester.test(x_test, y_test)
accuracy = round(accuracy * 10, 2)

np.savez(f'models/model_{accuracy}_{datetime.now().strftime("%Y-%m-%d_%H-%M")}.npz',
         dense1_weights=dense1.weights,
         dense1_biases=dense1.biases,
         dense2_weights=dense2.weights,
         dense2_biases=dense2.biases,
         dense3_weights=dense3.weights,
         dense3_biases=dense3.biases
         )
