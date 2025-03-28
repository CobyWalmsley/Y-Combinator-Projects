import numpy as np
import random
##I'm just going to copy the original one to see if I can get regular learning
def sigmoid(z):
        return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1-sigmoid(z))






number_list = [1,2,3,4,5,6,7,8,9,10,11]
def tuple_to_array(tuple1):
    question = np.array(tuple1[0])
    answer = np.array(tuple1[1])
    final = (question, answer)
    return(final)

def unpack_file(filenumber): #convert a training file into a list of arrays
    text_file_path = f"C:/Users/cobyw/OneDrive/Documents/Machine_Learning/Stock_Prices/Total_training_data/training_list{filenumber}.txt"
    tuple_list = []
    with open(text_file_path, "r") as f:
        lines = f.readlines()
        total_list = lines[0]
        total_list = eval(total_list)
        random.shuffle(total_list)
        for point in total_list:
            tuple_list.append(tuple_to_array(point))
    return(tuple_list)



class Psychic_but_real(object):
##I dont even think this network includes a regularization parameter
    def __init__(self,sizes): #The list sizes is the shape of the network. [Mine will be 420, hidden layers, 3]
        self.num_layer = (len(sizes))
        self.sizes = sizes
        self.biases = [np.random.randn(y,1)for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]  #We have now created a network with the architecture we need, filled with random weights and biases.

    
    def feedforward(self,a):
        for b,w in zip(self.biases,self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a

    def SGD(self, training_data,epochs,mini_batch_size,eta, test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print(f"Epoch{j}: {self.evaluate(test_data)}/{n_test}")
            else:
                print(f"Epoch {j} complete")

    def update_mini_batch(self, mini_batch,eta):
          

