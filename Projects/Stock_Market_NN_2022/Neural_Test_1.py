##practicing the Network Class
import numpy as np
import random
class MyNetwork(object):
    def __init__(self,sizes):
        self.num_layers = len(sizes) ##create a variable called num_layers that shows how many layers the neural net has
        self.sizes = sizes #create a variable that is the same as the list you inputted as an argument
        self.biases = [np.random.randn(y,1) for y in sizes[1:]] #I think this creates a numpy array of random integers from -1 to 1 that is the same size as the network that will be used as the weights
        #the first layer doesn't need biases because it is an input layer
        self.weights = [np.random.randn(y, x) 
                        for x, y in zip(sizes[:-1], sizes[1:])] 

        def sigmoid(z): #estalblishes what the sigmoid function is
            return(1.0/1.0+np.exp(-z))

        def sigmoid_prime(z):
            return sigmoid(z)*(1-sigmoid(z))

        def cost_derivative(self, output_activations, y):
            return (output_activations-y)



        def feedforward(self, a): #return the network output if a is the input
            for b,w in zip(self.biases, self.weights):
                a =  sigmoid(np.dot(w,a)+b)
                return a
        ## I understanf enrything from here up. Now we have to get it to learn.

        def SGD(self,training_data, epochs, mini_batch_size, eta, test_data=None)  #the training data is a list of tuples (x,y) representing training inputs and their desired outputs.
            if test_data: n_test=len(test_data) # how large is the test data set
            n = len(training_data) #how large is the training data set)
            for j in xrange(epochs):
                random.shuffle(training_data)
                mini_batches = [
                    training_data[k:k+mini_batch_size]
                    for k in xrange(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta)
            if test_data:
                print( f"Epoch {j} {self.evaluate(test_data)} / {n_test}")
                   
            else:
                print( "Epoch {0} complete".format(j))

        def backprop(x,y): #returns a tuple (nabla_b, nabla_w) representing the gradient for the cost function C_x. They are lists of numpy arrays.
            nabla_b = [np.zeros(b.shape) for b in self.biases] #create a numpy array of the same size and shape as self.biases, except filled with zeros
            nabla_w = (np.zeros(w.shape) for w in self.weights)
            activation = x
            activations = [x] #list to store all the activations, layer by layer
            zs = [] #empty list to store z vectors, layer by layer
            for b,w in zip(self.biases,self.weights):
                z = np.dot(w, activation)+b
                zs.append(z)
                activation = sigmoid(z)
                activations.append(activation)  #take the weights and biases and return an activation
            delta = (self.cost_derivitave(activations[-1],y) * sigmoid_prime(zs[-1]))
            nabla_b[-1] = delta
            nabla_w[-1] = np.dot(delta, activations[-2].transpose())

            for l in xrange(2, self.num_layers):
                z = zs[-1]
                sp = sigmoid_prime(z)
                delta = np.dot(self.weights[-1+1].transpose(),delta)*sp
                nabla_b[-1] = delta
                nabla_w[-1] = np.dot(delta, activations[-l-1].transpose())
            return(nabla_b,nabla_w)

        def update_mini_batches(self, mini_batch, eta): #updates the mni batches using stochastic gradient descent and backpropogation
            nabla_b = [np.zeros(b.shape) for b in self.biases]
            nabla_w = [np.zeros(b.shape) for b in self.weights] #empty numpy arrays that are the same shape and size as the weights and bias arrays: designed to hold future gradient values
            delta_nabla_b, delta_nabla_w = self.backprop(x,y) #we haven't made backprop yet, but this is where the laerning happens it seems
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)] # add the updated values to the gradient vector for the biases.
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)] #same for weights
            self.weights = [w-eta/len(mini_batch)*nw for w, nw in zip (self.weights, nabla_w)] # subject each mini batch to the quadratic cost function
            self.biases = [b-eta/len(mini_batch)*nb for b, nb in zip(self.weights, nabla_w)] # now we have updated weights and biases

       
