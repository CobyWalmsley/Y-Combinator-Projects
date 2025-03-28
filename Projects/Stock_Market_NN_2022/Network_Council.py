import numpy as np
import random
import json
import sys

class CrossEntropyCost(object):  #cross entropy cost function
    def fn(a, y):
        #Return the cost associated with an output ``a`` and desired output y
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
    
    def delta(z, a, y): #return the error vector delta from the last layer

        return (a-y)

class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost): #the list "sizes " contains the number of neurons in each layer of the neural network
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost=cost

    def default_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                    for x, y in zip(self.sizes[:-1], self.sizes[1:])]

  #good up to here      

    def feedforward(self, a): #return the output of a network if the vector a is the input
        answer1=(np.dot(self.weights[0],(np.array(a)).reshape((419,1))))+self.biases[0]
        answer2 = np.dot(self.weights[1],answer1,)+self.biases[1]
        return answer2



    def SGD(self, training_data_file_list, epochs, mini_batch_size, eta, lmbda = 0.0,
            evaluation_data_file=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):

        if evaluation_data_file: n_data = 5691
        n = 62601 #length of training data
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            random.shuffle(training_data_file_list)
            for k in training_data_file_list:
                file_path = f"C:/Users/cobyw/OneDrive/Documents/Machine_Learning/Stock_Prices/Total_training_data/training_list{k}.txt"
                with open(file_path, "r") as f:
                    lines = f.readlines()
                    day_list = lines[0]
                    f.close()
                training_data = eval(day_list)
                random.shuffle(training_data)
                mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, 5691, mini_batch_size)]
                for mini_batch in mini_batches:
                    self.update_mini_batch(
                        mini_batch, eta, lmbda, 5691)
            print(f"                           Epoch {j} training complete")
            if monitor_training_cost:
                cost = self.total_cost(training_data_file_list, lmbda)
                training_cost.append(cost)
                print(f"cost on training data: {cost}")
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data_file_list, convert=True)
                total_accuracy = accuracy[0]
                positive_accuracy = accuracy[1]
                negative_accuracy = accuracy[2]
                neutral_accuracy = accuracy[3]
                training_accuracy.append(accuracy[0])
                print(f"Accuracy on training data: {total_accuracy} / 62601") 
                print(f"Buy Point Accuracy: {positive_accuracy}/3900")
                print(f"Sell Point Accuracy:{negative_accuracy}/3562")
                print(f"        Neutral Accuracy:{neutral_accuracy}/49448")
                print(f"        Percent Buy Guesses: {positive_accuracy}/ {accuracy[4]}, ({positive_accuracy / (accuracy[4]+.01)})")
                print(f"        Percent Sell Guesses: {negative_accuracy}/ {accuracy[5]}")
                print(f"        Percent Neutral Guesses: {neutral_accuracy}/ {accuracy[6]}")
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data_file, lmbda, convert=True)
                evaluation_cost.append(cost)
                print(f"Cost on evaluation data: {cost}")
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data_file)
                total_accuracy = accuracy[0]
                positive_accuracy = accuracy[1]
                negative_accuracy = accuracy[2]
                neutral_accuracy = accuracy[3]
                training_accuracy.append(accuracy)
                print(f"Accuracy on training data: {total_accuracy}/5691") 
                print(f"Buy Point Accuracy: {positive_accuracy}")
                print(f"Sell Point Accuracy:{negative_accuracy}")
                print(f"Neutral Accuracy:{neutral_accuracy}")
                print(f"        Percent Buy Guesses: {positive_accuracy}/ {accuracy[4]}, ({positive_accuracy / (accuracy[4]+.01)})")
                print(f"        Percent Sell Guesses: {negative_accuracy}/ {accuracy[5]}")
                print(f"        Percent Neutral Guesses: {neutral_accuracy}/ {accuracy[6]}")
                evaluation_accuracy.append(accuracy[0])
                print(f"Accuracy on evaluation data: {self.accuracy(evaluation_data_file)} / {n_data}")
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for k, o in mini_batch:
            x=(np.array(k))
            x=x.reshape((419,1))
            y=(np.array(o))
            y=y.reshape((3,1))
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x #the 419 input value
        activations = []
        activations.append(activation)
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = (np.dot(w, activation)+b)
            zs.append(z)
        
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def accuracy(self, data_files, convert=False, evaluation=False):
        if type(data_files==list):
            results_list = []
            positive_sum=0
            negative_sum=0
            neutral_sum=0
            buy_total=0
            sell_total=0
            neutral_total=0
            for n in data_files:
                file_path = f"C:/Users/cobyw/OneDrive/Documents/Machine_Learning/Stock_Prices/Total_training_data/training_list{n}.txt"
                with open(file_path, "r") as f:
                    lines = f.readlines()
                    day_list = lines[0]
                    f.close()
                day_list=eval(day_list)
                results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for x,y in day_list]
                results_list=results_list+results
                  #why does it just guess zero every time
                
            for x,y in results_list:
                if int(x==y):
                    if x==0:
                        negative_sum +=1
                    if x==1:
                        neutral_sum+=1
                    if x==2:
                        positive_sum+=1
                if x==2:
                    buy_total+=1
                if x==0:
                    sell_total+=1
                if x==1:
                    neutral_total+=1
        return ((sum(int(x == y) for (x, y) in results_list)),positive_sum,negative_sum,neutral_sum,buy_total,sell_total,neutral_total)     

    def total_cost(self, data_files, lmbda, convert=False,evaluation=False):
        cost = 0.0
        if type(data_files)==int:
            file_path = f"C:/Users/cobyw/OneDrive/Documents/Machine_Learning/Stock_Prices/Total_training_data/training_list{data_files}.txt"
            with open(file_path, "r") as f:
                lines = f.readlines()
                day_list = lines[0]
                f.close()
            day_list=eval(day_list)
            for h,i in day_list:
                print("test",h)
                x=(np.array(h))
                print(x)
                x=x.reshape((419,1))
                print(x)
                y=(np.array(i))
                y=y.reshape((3,1))
                a = self.feedforward(x)   ##the problem is this variable doesn't exist. 
                cost += self.cost.fn(a, y)/5691
            cost += 0.5*(lmbda/5691)*sum(
                np.linalg.norm(w)**2 for w in self.weights)
            return cost
        else:
            for n in data_files:
                file_path = f"C:/Users/cobyw/OneDrive/Documents/Machine_Learning/Stock_Prices/Total_training_data/training_list{n}.txt"
                with open(file_path, "r") as f:
                    lines = f.readlines()
                    day_list = lines[0]
                    f.close()
                day_list=eval(day_list)
                for x,y in day_list:
                    x=(np.array(x))
                    x=x.reshape((419,1))
                    y=(np.array(y))
                    y=y.reshape((3,1))
                    a = self.feedforward(x)
                
                    cost += self.cost.fn(a, y)/5691
                cost += 0.5*(lmbda/5691)*sum(
                    np.linalg.norm(w)**2 for w in self.weights)
            return cost

    def save(self, filename):
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def load(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

##Import 10 trained networks and have them vote.
#Import 10 networks and make them functions (what does that even mean) and run the data point through all 10. 
#Take their collective opinion as the real opinion and check if its accurate for the entire data set. Compare that to the accuracy of each network in turn.

#def test_net(net_name)

#open the file with questions and answers
#run the first network through every point. Put the guesses in a list. Add the indices of every buy point to a separate list
#do this with all the networks, getting the accuracy of each network each time.
#check how many of each number there are to see how many votes every buy point got.
#check if these points are actually correct by checking if the question at the index of the answer is a buy.
#total these up and get a percent, and see if it beats the best network.

def multi_runthrough(filenames, net_file_name):
    guess_index = 0
    buy_count = 0
    correct_buys = 0
    buy_guesses = []
    correct_buy_list = []

    net= load(net_file_name)
    for file in filenames:
        file_path = f"C:/Users/cobyw/OneDrive/Documents/Machine_Learning/Stock_Prices/Total_training_data/training_list{file}.txt"
        with open(file_path, "r") as f:
            lines = f.readlines()
            day_list = lines[0]
            f.close()
        day_list=eval(day_list)
        guess_list = [(np.argmax(net.feedforward(x)), np.argmax(y)) for x,y in day_list]      
        for x,y in guess_list:
            guess_index+=1
            if x==2:
                buy_guesses.append(guess_index)
                buy_count+=1
                if x==y:
                    correct_buys +=1
                    correct_buy_list.append(guess_index)
        print(f"{correct_buys} / {buy_count}")
    return(buy_guesses,correct_buy_list,correct_buys,buy_count)

##We have horrible problems

def comparator(day_list,filenames):
    total_buy_guesses = []
    total_correct_buys = []
    new_list = []
    overlaps = []
    for name in filenames:
        guesses = multi_runthrough(day_list,name)
        buy_guesses = guesses[0]
        correct_buys = guesses[1]
        total_correct_buys += correct_buys
        total_buy_guesses += buy_guesses
    for i in buy_guesses:
        if buy_guesses.count(i) >=2:
            overlaps.append(i)
    total_correct = 0
    
    for i in overlaps:
        if i in total_correct_buys:
            total_correct +=1
        
    print(f'{total_correct} / {len(overlaps)}')

#multi_runthrough([1,2,3,4,5,6,7,8,9,10,11],"test_network3.txt")

#comparator([1,2,3,4,5,6,7,8,9,10,11], ["test_network1.txt","test_network2.txt","test_network3.txt","test_network4.txt","test_network5.txt","test_network6.txt","test_network7.txt","test_network8.txt"])

comparator([12,13,14,15,16,17,18,19,20,21,22,23],["test_network2.txt"])
    #check if there are any duplicates in the buy guesses, then check if they're right.
#What percent would you have made

#tn1 = 71/99 (71%) (32%)
# tn2 = (53/111) (47.7%) Not worth it
#tn3 = 207/264 (78.4%) (22.8% on month 2)
#tn4 = 682/1038 (65.7%)
#tn5 = 72%
#tn6 = 19/27 (70%)
#tn 7= 38/58 (65.5%)
#tn 8 = 43/66 (65.1%)


#testnetwork1 is fairly accurate and testnetwork2 is not very accurate at all.   
#No overlaps after 8 networks. Make 20 more. Make a program to make them in hours.
#

