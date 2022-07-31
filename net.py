import numpy as np

#variables
LEARNING_RATE = 0.5




def print_net(layers, num_layers):
    print()
    print("input layer:\t", layers[0])
    for i in range(1, num_layers -1):
        print("hidden layer:\t", layers[i])
    print("output layer:\t", layers[-1])
    print()

def forward_propogation(num_layers):
    print("starting forward propogation ...")
    for i in range(num_layers-1):
        w = weights[i]
        x = layers[i]
        b = biases[i]
        z = np.add(np.dot(w, x), b)
        a = sigmoid(z)
        layers[i+1] = a

def sigmoid(z):
    return 1 /(1 + np.exp(-z))
 
def back_propogation(error): #d out y1 = y1  dy1 = dy2
    dE_dY1 = - np.subtract(expected_output, layers[-1])
    dY1_dY2 = 0
    dY2_dW = 0
    dE_dW = dE_dY1 * dY1_dY2 * dY2_dW
    update_weights(dE_dW)
    return 0

def update_weights(dE_dW):
    dE_dW = np.dot(LEARNING_RATE, dE_dW)
    layers = np.subtract(layers, dE_dW)

def cost_function(expected_output):
    return np.sum(1/2 * (layers[-1] - expected_output) ** 2)

def print_results():
    print(layers[0], "  =  ", layers[-1])
    print()

def count_num_neurons():
    num = 0
    for layer in layers:
        for element in layer:
            num += 1
    return num

#XOR data
training_inputs = [[1,0], [1, 1], [0, 0]]
training_outputs = [[1],[0],[0]]
testing_input = [0,1]
testing_output = [1]

input_layer = training_inputs[0]
output_layer = training_outputs[0]

# for i in range(len(training_inputs)):
#     input_layer = training_inputs[i]
#     expected_output = training_outputs[i]
#     layers[0] = input_layer
#     forward_propogation()
#     cost += cost_function(expected_output)
#     print_results()
# back_propogation()

input_layer = [0.05, 0.10]
weights = [[[0.15, 0.20], [0.25, 0.30]], 
           [[0.40, 0.45], [0.50, 0.55]]]
biases = [0.35, 0.60]
expected_output = [0.01, 0.99]
layers = []
hidden_layers = [[0, 0]]
output_layer = [0, 0]
layers.append(input_layer)
for l in hidden_layers:
    layers.append(l)
layers.append(output_layer)

num_neurons = count_num_neurons()
num_layers = len(layers)

forward_propogation(num_layers)
print_net(layers, num_layers)
error = cost_function(expected_output)
back_propogation(error)
print_results()

