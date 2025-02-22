import numpy as np
import matplotlib.pyplot as plt
import os
from numba import jit, prange

os.environ["OMP_NUM_THREADS"] = "8"

@jit
def fast_dot(inputs, weights, biases):
    return np.dot(inputs.astype(np.float64), weights) + biases

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, weights=None, biases=None):
        if weights is None:
            self.weights = np.random.randn(n_inputs, n_neurons)
        else:
            self.weights = weights

        if biases is None:
            self.biases = np.zeros((1, n_neurons))
        else:
            self.biases = biases

        self.output = np.zeros((n_inputs, n_neurons))
    
    def forward(self, inputs):
        self.inputs = inputs
        self.output = fast_dot(inputs, self.weights, self.biases)

    def backward(self, dvalues, alpha):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        self.dinputs = np.dot(dvalues, self.weights.T)

        self.weights -= alpha * self.dweights
        self.biases -= alpha * self.dbiases

class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    def forward(self, inputs):
        exp = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp / np.sum(exp, axis=1, keepdims=True)

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        correct_confidences = np.sum(y_pred_clipped * y, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
    def backward(self, dvalues, y):
        self.dinputs = (dvalues - y)
        
class TestModel:
    def __init__(self, model, n_inputs, n_neurons):
        self.dense_layers = []
        for i in range(len(n_inputs)):
            self.dense_layers.append(Layer_Dense(n_inputs[i], n_neurons[i], model[i][0], model[i][1]))
        self.activation1 = Activation_ReLU()
        self.activation2 = Activation_Softmax()
    
    def forward(self, inputs):
        self.dense_layers[0].forward(inputs)
        output = self.dense_layers[0].output
        for layer in self.dense_layers[1:]:
            self.activation1.forward(output)
            layer.forward(self.activation1.output)
            output = layer.output

        self.activation2.forward(output)
        return self.activation2.output
    
    def predict(self, pdistr, y):
        predictions = np.argmax(pdistr, axis=1)
        return predictions, y, predictions == y
    
    def test(self, X_test, y):
        out = self.forward(X_test)
        predictions, truth, correct = self.predict(out, y)
            
        accuracy = sum(correct) / len(correct)

        print(f'Model accuracy: {accuracy}')
        print('Testing complete')

        return accuracy

class TestMNIST(TestModel):
    def disp(self, X_test, y):
        out = self.forward(X_test)
        predictions, truth, correct = self.predict(out, y)

        for i in range(len(predictions)):
            plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
            plt.title(f"Prediction: {predictions[i]}, Ground Truth: {truth[i]}")
            plt.show()
