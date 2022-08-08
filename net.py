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
    for i in range(num_layers-1):
        w = weights[i]
        x = layers[i]
        b = biases[i]
        z = np.add(np.dot(w, x), b)
        a = sigmoid(z)
        layers[i+1] = a

def sigmoid(z, derive = False):
    if(derive):
        return z(1-z)
    return 1 /(1 + np.exp(-z))
 
def back_propogation(cost_gradient, i): 
    print(cost_gradient)
    print(layers[i])
    weight_gradient = np.dot(cost_gradient, np.transpose(layers[i]))
    update_weights(weight_gradient, i)
    bias_gradient = cost_gradient
    update_biases(bias_gradient, i)
    return np.dot(np.transpose(weights[i]), cost_gradient)

def update_weights(gradient, i):
    gradient = np.dot(LEARNING_RATE, gradient)
    weights[i] = np.subtract(weights[i], gradient)

def update_biases(gradient, i):
    gradient = np.dot(LEARNING_RATE, gradient)
    biases[i] = np.subtract(biases[i], gradient)


def cost_function(expected_output):
    return np.power(np.subtract(layers[-1], expected_output),1)

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


# input_layer = [0.05, 0.10]
weights = [[[0.15, 0.20], [0.25, 0.30]], 
           [[0.40, 0.45]]]
biases = [0.35, 0.60]
# expected_output = [0.01, 0.99]


#training
for k in range(len(training_inputs)):
    layers = []
    hidden_layers = [[0, 0]]
    output_layer = [0]

    input_layer = training_inputs[k]
    expected_output = training_outputs[k]

    layers.append(input_layer)
    for l in hidden_layers:
        layers.append(l)
    layers.append(output_layer)

    num_neurons = count_num_neurons()
    num_layers = len(layers)
    for x in range(2):
        forward_propogation(num_layers)
        cost_gradient = cost_function(expected_output)
        for i in range(num_layers-2, -1, -1):
            cost_gradient = back_propogation(cost_gradient, i)

    



#testing
input_layer = testing_input
expected_output = testing_output
hidden_layers = [[0, 0]]
output_layer = [0]
layers = []
layers.append(input_layer)
for l in hidden_layers:
    layers.append(l)
layers.append(output_layer)
forward_propogation(num_layers)
print_net(layers, num_layers)

