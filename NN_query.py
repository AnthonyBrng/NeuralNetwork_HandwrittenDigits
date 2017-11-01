from NeuralNetwork import NeuralNetwork
import numpy


#Parameters
inodes = 784
hnodes = 300
onodes = 10
learning_rate = 0.3

# initializing
Anton = NeuralNetwork(inodes, hnodes, onodes, learning_rate)

#loading data
training_file = open("mnist_testdata/mnist_train_60k.csv")
training_datalist = training_file.readlines()
training_file.close()

# train the network
for data in training_datalist:

    data_values = data.split(',')
    scaled_input = (numpy.asfarray(data_values[1:]) /255.0 * 0.99) + 0.01
    target = numpy.zeros(onodes) + 0.01
    target[int(data_values[0])] = 0.99

    Anton.train(scaled_input, target)

    pass
# Query Anton
test_file = open("mnist_testdata/mnist_test_10k.csv")
test_datalist = test_file.readlines()
test_file.close()

scoreboard = numpy.zeros(len(test_datalist))
i = 0

# test the network
for data in test_datalist:

    data_values = data.split(',')
    scaled_input = (numpy.asfarray(data_values[1:]) /255.0 * 0.99) + 0.01

    answer = Anton.query(scaled_input)

    if (int(data_values[0])==numpy.argmax(answer)):
        scoreboard[i] = 1
    else:
        scoreboard[i] = 0
        pass

    i+=1

    pass

print("Anton hat ", numpy.sum(scoreboard) , " von ", len(test_datalist), "richtig")
print("Das entpsricht " , numpy.sum(scoreboard)/float(len(test_datalist))*100, "%" )
