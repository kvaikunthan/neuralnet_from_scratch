import numpy as np
import matplotlib.pyplot as plt

def create_data(N, K):
    D = 2 # Dimensionality
    X = np.zeros((N * K, D))
    Y = np.zeros(N * K, dtype='uint8')

    for i in range(K):
        ix = range(N * i, N * (i + 1))
        r = np.linspace(0.0,1,N) # radius
        t = np.linspace(i*4,(i+1)*4,N) + np.random.randn(N)*0.2
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = i
    
    return X, Y

def plot_decision_boundary(X, Y, model):
    # Create a mesh grid of points
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # Stack grid points into (N, 2) shape
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Forward pass through the model
    model[0].forward(grid_points)   # Dense layer 1
    model[1].forward(model[0].output)  # ReLU activation
    model[2].forward(model[1].output)  # Dense layer 2
    model[3].forward(model[2].output)  # Softmax activation

    # Get class predictions
    predictions = np.argmax(model[3].output, axis=1)
    
    # Reshape predictions to match mesh grid shape
    predictions = predictions.reshape(xx.shape)

    # Plot decision boundary
    plt.contourf(xx, yy, predictions, alpha=0.3, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Spectral, edgecolors='k')
    plt.title("Decision Boundary")
    plt.show()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.output = np.zeros((n_inputs, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

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
        correct_confidences = y_pred_clipped[np.arange(len(y_pred_clipped)), y]
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
    def backward(self, dvalues, y):
        samples = len(dvalues)
        y_one_hot = np.zeros_like(dvalues)
        y_one_hot[np.arange(samples), y] = 1
        self.dinputs = (dvalues - y_one_hot) / samples

X, Y = create_data(100, 3)

dense1 = Layer_Dense(2, 100)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(100, 3)
activation2 = Activation_Softmax()
loss_function = Loss_CategoricalCrossEntropy()

epochs = 10000
alpha = 0.1

for epoch in range(epochs):
    # Forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    # Compute loss
    loss = loss_function.calculate(activation2.output, Y)

    # Backward pass
    loss_function.backward(activation2.output, Y)
    dense2.backward(loss_function.dinputs, alpha)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs, alpha)

    if epoch % 100 == 0:
        predictions = np.argmax(activation2.output, axis=1)
        print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {np.mean(predictions == Y):.4f}")

plot_decision_boundary(X, Y, [dense1, activation1, dense2, activation2])