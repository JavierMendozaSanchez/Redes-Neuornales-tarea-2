import random 
import numpy as np
import matplotlib.pyplot as plt

class CrossEntropyCost:
    @staticmethod
    def fn(a, y):
        """En esta modificacion de codigo de cross-entrpy mantengo
        la funcion"""
        return np.sum(np.nan_to_num(-y*np.log(a) - (1-y)*np.log(1-a)))

class Network:
#ahora si implemento la incializacion de pesos de 1 sobre la raiz del numero
#de neruronas de entrada 
    def __init__(self, sizes):
      
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [
            np.random.randn(y, x) / np.sqrt(x)
            for x, y in zip(sizes[:-1], sizes[1:])
        ]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        training_data = list(training_data)
        training_data = training_data[:10000]   # usamos 10000 para compararlos
        #con los del tes
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        # En esta parte tambien añadimos la implementacion de guardar el costp
        #por cada epoca esto para el caso de el entrenamiento
        costs_train, costs_test = [], []

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            # Precisión y costo en entrenamiento
            accuracy_train = self.evaluate([(x, np.argmax(y)) for x, y in training_data])
            cost_train = self.total_cost(training_data)
            costs_train.append(cost_train)
#en este caso  hacemos lo mismo pero para el test entrenamiento y costo
            if test_data:
                accuracy_test = self.evaluate(test_data)
                cost_test = self.total_cost([(x, self.one_hot(y, self.sizes[-1])) for x, y in test_data])
                costs_test.append(cost_test)
                #se añade la opcion de imprimir estos valores
                print(f"Epoch {j}: Precisión entrenamiento {accuracy_train}/{n}, Costo entrenamiento: {cost_train:.4f}")
                print(f"Test: Precisión {accuracy_test}/{n_test}, Costo test: {cost_test:.4f}\n")
            else:
                print(f"Epoch {j}: Precisión {accuracy_train}/{n}, Costo entrenamiento: {cost_train:.4f}")

        # Graficar costo vs época de ambos tanto de test como de entrenamiento
        plt.plot(costs_train, label="Entrenamiento")
        if test_data:
            plt.plot(costs_test, label="Test")
        plt.xlabel("Epochs")
        plt.ylabel("Costo")
        plt.legend()
        plt.title("Costo vs Epochs")
        plt.show()

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # Promedio de gradientes
        nabla_w = [nw / len(mini_batch) for nw in nabla_w]
        nabla_b = [nb / len(mini_batch) for nb in nabla_b]

        # Actualización estándar pues aqui ya no estoy usando lo de momento
        self.weights = [w - eta * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - eta * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward usando Cross-Entropy t la sigmoid
        delta = (activations[-1] - y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].T)

        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def total_cost(self, data):
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            cost += CrossEntropyCost.fn(a, y)
        return cost / len(data)

    @staticmethod
    def one_hot(y, num_classes):
        e = np.zeros((num_classes, 1))
        e[y] = 1.0
        return e

#### Funciones auxiliares
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))
