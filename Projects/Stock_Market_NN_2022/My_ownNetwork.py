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
                print(f"Accuracy on training data: {total_accuracy} / 119490") 
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
                print(f"Accuracy on evaluation data: {total_accuracy}/11382") 
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

net = Network([419,150, 3], cost=CrossEntropyCost)
training_files = [1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21,22]
net.SGD(training_files, 15, 70, 0.15, lmbda = 2 ,evaluation_data_file=[11,23],monitor_evaluation_accuracy=True,monitor_evaluation_cost=False,monitor_training_accuracy=True,
monitor_training_cost=False)

net.save("test_network27.txt")


## Spend another day on hyper-parameters. 

## Train 10 neural nets for 1 Epoch and either listen to all of them or make them take a vote.

#I need to monitor this code to show me how many buy guesses it makes, to see how safe this would be to use
#If it guess a ton of wrong buys, its dangerous, but if its only guessing a few and getting them all right, I'm done.


#Trial 1: 100, 40, 0.3, 5.0: Neutral near perfect, Positive and negative near zero
#Trial 2: 100, 40, 0.3, 10: Neutral: 40428 Buy: 1877 Sell:1055. 
#Trial 3: 100,40, 0.3, 15: TP: E25 T=48436 U=874 D=1613 EP: E20 T=4719 U=98 D=170
#Trial 4: 100, 40, 0.3, 20: Best Epoch 21: T = 48948, U = 987, D = 1189 Eval: T=4858 U=87 D=112
#Trial 5: 100, 40, 0.6, 20: Terrible, way too high of learning rate
#Trial 6: 150,40,0.3, 20: Learns backwards. Lowering Learning rate
#Trial 7: 150, 40, 0.15, 20 Learns backwards: Lowering Learning rate
#Trial 8: 150, 40, 0.08, 10 About the best so far. T= 49786 N: 47924 U: 850 D: 1012... Eval: 4947 U: 84 D: 94 Epoch 15
#Trial 9: 150, 40, 0.1, 5: Starts overfitting at epoch 16
#Trial 10: 150, 100, 0.1, 8 Gets up high and starts learning backwards
#Trial 11: 150, 100, 0.06, 9.0 Non-linear learning. Displayed buying accuracy around 70%. Going to retry earlier successes.
#Trial 12: Have discovered that when it guess a lot of buys correctly it also gets a lot wrong. Keep number low. Up mini batch size
#Trial 13: 150, 150, 0.1, 10: Very accurate buying percentage in Epoch 3 and then tries to guess more buys and gets them wrong. Lowering learning rate. Buy accuracy does stay over 50%
#Trial 14:  150, 150, 0.02, 12. Training for 80 Epochs with tiny learning rate to see what happens. It will never buy or sell.
#Trial 15: 150,150, 0.08, 10: When the mini-batch size is too high, it never guesses to buy or sell.
#Trial 16: 150, 50, 0.2, 15: Buy accuracy starts bad.
#Trial 17: 150, 60, 0.1, 5: Peak buy accuracy on Epoch 3. at 73%. Will try learning
#Trial 18: 150, 60, 0.1, 12: Lambda too high. Peak 44%
#Trial 19: 150,60,0.1, 5 Did the same thing twice accidentally. Peak accuracy 66%
#Trial 20:  150, 60, .15, 5. Buys stayed low. Epoch 2: 75% and 91% (11/12) Epoch 3 75% and 67% (41/61)
#Trial 21: 150 60, .15, 3. I'm going to only count ones with similar accuracies in training and eval. Epoch 3: 71% and 67%
#Trial 22: 150, 70, .15, 3 Epoch 3: 69% and 67%
#Trial 23: 150, 70, .15, 2 Epoch 2: 77% and 75% (24/32)
#Trial 24: 150, 70, .15, 1.5 Epoch 2: 72% and 83.5% (19/23)

#I now have twice the amount of data







## 10 Trials with copy of Trial 23: (69%,63%) (74%, 68%) (74%, 65%) (71%, 66%) (79%,78%)  73.4% average  68% average



#New plan: Make 10 networks and make them vote.
#make a voting system


#Trial 2 Notes: Peaked on final Epoch 15. Worth retrying to 20 or 25
#Trial 3 Notes: Overfitting is clear in this Epoch. Up the lambda. Better results overall. Highest TD on Epoch 20 as an outlier, but linear growth up through Epoch 25.
#Highest ED is in Epoch 19. Part of me wants to train for 50 Epochs to see. Definite learning happening.
#Trial 4 Notes: The learning is stopping now. I'm going to try upping the learning rate. The eval accuracy and training accuracy is capping off and fluctuating
#Trial 5 Notes: Terrible, way too high of learning rate
#Trial 6 Nots: Learns backwards. Lowering Learning rate
#Trial 7: 150, 40, 0.15, 20 Learns backwards: Lowering Learning rate
#Trial 8: 150, 40, 0.08, 20 Learns backwards. Lowering learning rate
#Trial 




#SGD(self, training_data_file_list, epochs, mini_batch_size, eta, lmbda = 0.0,
            #evaluation_data_file=None,
            #monitor_evaluation_cost=False,
            #monitor_evaluation_accuracy=False,
            #monitor_training_cost=False,
            #monitor_training_accuracy=False):
