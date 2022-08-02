import numpy as np

#variables
LEARNING_RATE = 0.1

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

def sigmoid(z, deriv=False):
    if (deriv):
        return z * (1-z)
    return 1 /(1 + np.exp(-z))
 
def back_propogation(cost_gradient, i): 
    weight_gradient = np.multiply(cost_gradient, np.transpose(layers[i]))
    update_weights(weight_gradient, i)
    update_biases(cost_gradient, i)
    return np.multiply(np.transpose(weights[i]), cost_gradient)

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

def init_net(testing_input, hidden_layers):
    layers = []
    input_layer = testing_input
    output_layer = [0]
    layers.append(input_layer)
    for l in hidden_layers:
        layers.append(l)
    layers.append(output_layer)
    return layers
    


if __name__ == "__main__":
    #XOR data
    training_inputs = [[1,0], [1, 1], [0, 0], [0,1]]
    training_outputs = [[1],[0],[0], [1]]

    hidden_layers = [[0, 0]]
    weights = [ [ [0.15, 0.20], [0.25, 0.30] ], 
                [0.40, 0.45]                  ]
    biases = [0.35, 0.60]

    #training
    for x in range(100):
        #initialize net
        k = np.random.randint(0, len(training_inputs))	
        layers = init_net(training_inputs[k], hidden_layers)
        expected_output = training_outputs[k]

        num_neurons = count_num_neurons()
        num_layers = len(layers)

        forward_propogation(num_layers)
        cost_gradient = cost_function(expected_output)
        for i in range(num_layers-2, -1, -1):
            cost_gradient = back_propogation(cost_gradient, i)

        

    #testing
    testing_input = [0,0]
    testing_output = [0]

    layers = init_net(testing_input, hidden_layers)
    forward_propogation(num_layers)
    print_net(layers, num_layers)
