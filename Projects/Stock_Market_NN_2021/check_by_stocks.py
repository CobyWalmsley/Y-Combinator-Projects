import random
import numpy as np
import json
import sys
from decimal import Decimal
from selenium import webdriver
import time
from datetime import datetime
import csv
results_list=[]
randlist=[]
from sp500 import sp500

results2_list=[]
class CrossEntropyCost(object):

        @staticmethod
        def fn(a, y):
            return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

        @staticmethod
        def delta(z, a, y):
            return (a-y)

class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.cost=cost

    def feedforward(self, a):
            
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)   
        return a

def load(filename):

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
for x in range(50):
    number=random.randint(1,450)
    randlist.append(number)
stock_to_check=[]
for integer in randlist:
    symbol=sp500[integer]
    stock_to_check.append(symbol) #string of the stock you want to see #1 or 0 for up or down
    outcome_up=1
    outcome_down=0
    stockdict2={}
    stockdict={}
    newday=[]

for trial in stock_to_check:
    PATH = r'C:/Users/cobyw/Documents/Important Documents/chromedriver.exe'
    driver = webdriver.Chrome(PATH)
    stock_url= f'https://finance.yahoo.com/quote/{trial}/history?period1=1586908800&period2=1644883200&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true'
    driver.get(stock_url) #Go to the stocks yahoo history page
    last_height=657
    for k in range(1,30): #scroll to the bottom
        last_height=last_height+1000
        driver.execute_script(f"window.scrollTo(0, {last_height});")
        time.sleep(.01)
    float_list=[]
    for j in range(1,350): #how many days plus a buffer for dividends
        res=[]
        if len(res)<298:
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

    up_list=product_list.copy()

    last_try=[]
    i=0
    for x in range(0,298):
        tomorrow=product_list[x]
        last_try.append(tomorrow)
        negative_day=last_try.copy()
        positive_day=last_try.copy()
        negative_day[:0] = [outcome_down]
        positive_day[:0] = [outcome_up]
        i=1

    if i==1: #how many days-1 ur hotv im not wearing a bra
        stockdict[trial]=negative_day
        stockdict2[trial]=positive_day

    point=[]
    point2=[]
    datalist=[]
    for key in stockdict:
        down_days=stockdict[key]
        for h in down_days:
            if h>0.5:
                h=1
            if h<=0.5:
                h=0
            point.append(h)
        datalist.append(point)
    
    empty_questions=[]
    for h in datalist:
        k=len(h)
        between_array2=np.array(h)
        question_array2 = between_array2.reshape((k,1))
        empty_questions.append(question_array2)
    
    result_list=[]
    for t in empty_questions:
        down_result=(net.feedforward(t))
        for d in down_result:
            new=d.tolist()
            result_list.append(new)
    print(result_list,'results')
    zero_spot=result_list[0]
    one_spot=result_list[1]


    if zero_spot>=one_spot:
        result1=(f'{trial} will go down tomorrow if the opening price is down,')
        results2_list.append(result1)
    if one_spot>zero_spot:
        result1=(f'{trial} will go up tomorrow if the opening price is down,')
        results2_list.append(result1)


    datalist2=[]
    for key in stockdict2:
        up_days=stockdict2[key]
        for h in up_days:
            if h>0.5:
                h=1
            if h<=0.5:
                h=0
            point2.append(h)   
        datalist2.append(point2)
    
    empty_questions=[]
    for h in datalist2:
        k=len(h)
        between_array2=np.array(h)
        question_array2 = between_array2.reshape((k,1))
        empty_questions.append(question_array2)
    result_list=[]
    for t in empty_questions:
        down_result=(net.feedforward(t))
        for d in down_result:
            new=d.tolist()
            result_list.append(new)
    print(result_list)
    zero_spot2=result_list[0]
    one_spot2=result_list[1]

    if zero_spot2>=one_spot2:
        result2=(f'{trial} will go down tomorrow if the opening price is up,')
        results2_list.append(result2)
    if one_spot2>zero_spot2:
        result2=(f'{trial} will go up tomorrow if the opening price is up,')
        results2_list.append(result2)
    print(result1, result2)
    driver.close()
    datalist.clear()
    stockdict.clear()
    stockdict2.clear()
    datalist2.clear()
    result_list.clear()


print(symbol)
print(results2_list)