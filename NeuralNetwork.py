import numpy
import scipy.special
import matplotlib.pyplot

# Classdefinition
# Neuralnetwork with 1 inputlayer, 1 hiddenlayer and 1 output layer
class NeuralNetwork:
    
    # initializing the neural network
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        
        # number of nodes in each layer & learningrate
        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes
        self.learn_rate = learning_rate
        self.activation_function = lambda x: scipy.special.expit(x)
        
        # weightmatrices-defintion between diffrent layers
        # randomvalues between - 1/squareroot(hnodes) AND + 1/squareroot(hnodes)
        # (change pow(i_nodes) to pow(h_nodes) maybe)
        self.wih = numpy.random.normal(0.0, pow(self.i_nodes,-0.5),(self.h_nodes,self.i_nodes))
        self.who = numpy.random.normal(0.0, pow(self.h_nodes,-0.5),(self.o_nodes,self.h_nodes))
        pass
    
    
    #train the network
    def train(self, inputs, targets):
        
        # transforms input_array into a 2dimensional array
        inputs = numpy.array(inputs,ndmin=2).T
        targets = numpy.array(targets, ndmin=2).T
        
        # calculating the output
        hidden_value = numpy.dot(self.wih, inputs) 
        hidden_output = self.activation_function(hidden_value)
        
        output_value = numpy.dot(self.who, hidden_output)
        final_output = self.activation_function(output_value)

        # calculating the error (output layer) (target - actual) 
        output_errors = targets - final_output
        
        #calculating error (hidden layer) 
        hidden_errors = numpy.dot(self.who.T,output_errors)
        
        #updating the weights
        self.who += self.learn_rate * numpy.dot(output_errors * final_output * (1.0 - final_output), hidden_output.T)
        self.wih += self.learn_rate * numpy.dot(hidden_errors * hidden_output * (1.0 - hidden_output), inputs.T)
        
        pass
    
    
    # Asking the network for an output
    def query(self, inputs):
        
        hidden_value = numpy.dot(self.wih, inputs) 
        hidden_output = self.activation_function(hidden_value)
        
        output_value = numpy.dot(self.who, hidden_output)
        final_output = self.activation_function(output_value)
        
        return final_output
        
    
    pass