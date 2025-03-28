import random
import numpy as np
import json
import sys
from decimal import Decimal
from selenium import webdriver
import time
from datetime import datetime
import csv
from sp500 import sp500

PATH = r'C:/Users/cobyw/Documents/Important Documents/chromedriver.exe'
driver = webdriver.Chrome(PATH)
stockdict={}
newday=[]
for x in range(3):
    rand=random.randint(1,485)
    stock=sp500[rand]
    newday.append(stock)

for trial in newday:
    stock_url= f'https://finance.yahoo.com/quote/{trial}/history?period1=1486598400&period2=1644192000&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true'
    driver.get(stock_url)
    last_height=657
    for k in range(1,30):
            last_height=last_height+1000
            driver.execute_script(f"window.scrollTo(0, {last_height});")
            time.sleep(.01)
    float_list=[]
    for j in range(1,350): #1764 how many days
        res=[]
        if len(res)<299:
            price_path=f'/html/body/div[1]/div/div/div[1]/div/div[3]/div[1]/div/div[2]/div/div/section/div[2]/table/tbody/tr[{j}]/td[2]/span'
            digit = driver.find_element_by_xpath(price_path)
            digit_text=digit.text
            float_list.append(digit_text)
            final=[]
            for c in float_list:
                if c!='Dividend':
                    if c!= 'Stock Split':
                        final.append(c)    
        
            for m in final:
                m=m.replace(',','')
                value=float(m)
                res.append(value)

    product_list=[]
    for n in range(len(res)-1): #how many days-(total/500)*12
        datasum=res[n+1]-res[n]
        data_divisor=datasum/res[n]
        round_data2=round(data_divisor,6)
        neuron_data=(0.5+round_data2)
        rounded_neuron=round(neuron_data,5)
        product_list.append(neuron_data)
    
    last_try=[]
    i=0
    for x in range(0,299):
        blah=product_list[x]
        last_try.append(blah)
    i=1

    if i==1: #how many days-1 
        stockdict[trial]=last_try

point=[]
datalist=[]
for key in stockdict:
    days=stockdict[key]
    print(key,days)
    for h in days:
            if h>0.5:
                h=1
            if h<=0.5:
                h=0
           
    datalist.append(days)

empty_questions=[]
for f in datalist:                  
    p=len(f)
    between_array=np.array(f)
    question_array = between_array.reshape((p,1))
    empty_questions.append(question_array)

class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.

        """
        return (a-y)


#### Main Network class
class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.cost=cost

    def feedforward(self, a):
        
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)   
        return a

def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

net=load("C:/Users/cobyw/vs/DeepLearningPython/better_network3")
net2=load("C:/Users/cobyw/vs/DeepLearningPython/better_network_3v2")

answer_list=[]


for t in empty_questions:
    print(net.feedforward(t))
    
    
    
    

    #switched=answer.reshape((60,1))
    #final_value=net2.feedforward(switched)
    #print(final_value)
    #answer_list.append(final_value)
print(newday)
print(answer_list)