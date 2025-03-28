import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

dates = np.array(["2/21/23","2/22/23","2/23/23","2/24/23","2/27/23","2/28/23","3/01/23","3/02/23","3/03/23"])
#25 is the max
stock_symbols = ["AAPL","BRK-B","MSFT","AMZN","GOOG","UNH","JNJ","XOM","JPM","NVDA","PG","GOOGL","V","HD","CVX","MA","LLY","PFE","ABBV","MRK","META","PEP","KO","BAC","AVGO","TMO","WMT","COST","CSCO","MCD","ABT","DHR","ACN","VZ","NEE","DIS","WFC","LIN","ADBE","PM","BMY","CMCSA","NKE","TXN","RTX","HON","COP","AMGN","CRM","T","NFLX","ORCL","IBM","UPS","UNP","SCHW","CAT","LOW","CVS","ELV","QCOM","GS","DE","SBUX","MS","SPGI","LMT","INTC","INTU","BA","GILD","BLK","PLD","MDT","AMD","AMT","ADP","CI","ISRG","GE","TJX","CB","MDLZ","C","AXP","PYPL","ADI","TMUS","AMAT","SYK","MMC","MO","DUK","BKNG","SO","NOC","NOW","REGN","PGR","VRTX","TGT"]
stock_symbols1 = ["GOOGL"]
#This is the 101 highest weighted companies in the S+p, including number 101, VRTX.
def create_files(day_list,stockname_list):

    for stockname in stockname_list:

        data=yf.download(f'{stockname}',start='2023-02-21', interval='5m',  end='2023-03-03',progress=False)[['Close']]
        data.head()
        string_data=data['Close'].tolist()
        
        rounded_data = list(np.around(np.array(string_data),2))
        length=len(rounded_data)

        segmented = ([rounded_data[x:x+78] for x in range(0, length, 78)])
        print(len(segmented))
        print(len(dates))
        for i in range(len(segmented)):
            segmented[i].append(day_list[i])
        segmented.pop()
        text_file_path = f"C:/Users/cobyw/OneDrive/Documents/Machine_Learning/Stock_Prices/Practice_test_data/{stockname}.txt"
        with open (text_file_path, "w") as f:
            f.write(f"{segmented}")
        print(stockname,"created")

todays_date = "1/3/23"
def update_days(today, stockname_list,start_date, end_date):   

    for stockname in stockname_list:
        data=yf.download(f'{stockname}',start= start_date, interval='5m',  end= end_date,progress=False)[['Close']]
        data.head()
        string_data=data['Close'].tolist()
        rounded_data = list(np.around(np.array(string_data),2))
        length=len(rounded_data)
        if length !=78: 
            print("appending length problem")
            rounded_data.pop()

        text_file_path = f"C:/Users/cobyw/OneDrive/Documents/Machine_Learning/Stock_Prices/Practice/{stockname}.txt"
        with open(text_file_path, "r") as f:
            lines = f.readlines()
            total_data_dict2 = lines[0]
            total_data_dict=eval(total_data_dict2)
            total_data_dict[today] = rounded_data
            f.close()
        with open (text_file_path, "w") as t:
            t.write(f"{total_data_dict}")    
            t.close()
            print(f"day added to {stockname}")

def check_if_stock_exists(stock_list):
    for stockname in stock_list:
        data=yf.download(f'{stockname}',start='2022-11-28', interval='5m',  end='2023-01-04',progress=False)[['Close']]
        data.head()
        print(f"{stockname} exists")

def percent_change(original,final):
    answer=(100*((final-original)/original))
    new_answer=round(answer,4)
    return(new_answer)


def change_to_percent_changes(training_example):
    new_data = []
    for i in range(len(training_example)-1):
        change = percent_change(training_example[i],training_example[i+1])
        new_data.append(change)
    return new_data

def tuple_switch(training_point,):
    answer = training_point.pop()
    if answer >=.6:
        answer_array= [1,0,0]
    if answer<=-.6:
        answer_array= [0,0,1]
    if -.6<answer<.6:
       answer_array= [0,1,0] 
    question_array = training_point 
    return((question_array,answer_array))

def create_training_points(long_list,len_question, dist_to_answer,len_of_day,points_per_day):
    training_points=[]
    for day_point in range(0,((len(long_list)-len_question-dist_to_answer)),len_of_day):
        for j in range(0,points_per_day): #for every day
            if (len(long_list)-len_question-dist_to_answer)<(day_point+j):
                print("data break")
                break
            else:
                training_point=[] #create an empty list to contain the point
                for n in range(0,len_question): #start making all points possible in the day
                    training_point.append(long_list[n+day_point+j])
                    answer = long_list[j+len_question+dist_to_answer+(day_point)-1] #find the answer to the question by adding the number of the day we're on plus the length of the question plus the distance to the answer plus the 
                    if n==(len_question-1):
                        training_point.append(answer)
                training_percents=change_to_percent_changes(training_point)
                list_tuple = tuple_switch(training_percents)
                training_points.append(list_tuple)
    return(training_points)

##length of day = 78, dist to answer = 12, points per day = 36, length of question = 420

def transform_to_training(stockname): #takes the text file and transforms it into another text file with all possible training data points
    text_file_path = f"C:/Users/cobyw/OneDrive/Documents/Machine_Learning/Stock_Prices/Practice_test_data/{stockname}.txt"
    with open(text_file_path, "r") as f:
            lines = f.readlines()
            day_list = lines[0]
            day_list = eval(day_list)
            for day in day_list:
                day.pop()
            combined_list = sum(day_list,[])
            total_symbol_data = create_training_points(combined_list, len_question = 420, len_of_day = 78, dist_to_answer = 12, points_per_day = 36)
            f.close()
    text_file_path2 = f"C:/Users/cobyw/OneDrive/Documents/Machine_Learning/Stock_Prices/Test_data_points/{stockname}data.txt"
    with open(text_file_path2, 'w') as t:
        t.write(f"{total_symbol_data}") 
        t.close()
        print(f"{stockname} data transformed")

def run_stats(stockname):
    text_file_path = f"C:/Users/cobyw/OneDrive/Documents/Machine_Learning/Stock_Prices/Training_by_stock_Month2/{stockname}.txt"
    with open(text_file_path, "r") as f:
            lines = f.readlines()
            day_list = lines[0]
            day_list = eval(day_list)
            positive_count = 0
            negative_count = 0
            neutral_count = 0
            for n in day_list:
                    if n[1][0]==1:
                        positive_count +=1
                    if n[1][1]==1:
                        neutral_count+=1
                    if n[1][2]==1:
                        negative_count+=1
            print("positives:",positive_count)
            print("negatives:",negative_count)
            print("neutrals:",neutral_count)

def transformer_king(symbol_list):
    count = 0
    for symbol in symbol_list:
        count+=1
        transform_to_training(symbol)
        print(f"{count}/{len(symbol_list)}")
#open every file and add it to a list
file_name_list = ["training_list12.txt","training_list13.txt","training_list14.txt","training_list15.txt","training_list16.txt","training_list17.txt","training_list18.txt","training_list19.txt","training_list20.txt","training_list21.txt","training_list22.txt","training_list23.txt","training_list24"]
def splitter(big_list_name, increment,file_list):
    text_file_path = f"C:/Users/cobyw/OneDrive/Documents/Machine_Learning/Stock_Prices/Training_by_stock_Month2/{big_list_name}"
    with open(text_file_path, "r") as f:
        lines = f.readlines()
        day_list = lines[0]
        day_list = eval(day_list)
    random.shuffle(day_list)
    for i in range(0,len(day_list),increment):
        list1 = day_list[i:i+increment]
        text_file_path = f"C:/Users/cobyw/OneDrive/Documents/Machine_Learning/Stock_Prices/Total_training_data/{file_list[int(i/increment)]}"
        with open(text_file_path, "w") as f:
            f.write(f"{list1}")
            f.close
            print("File created",len(list1))


def file_reader(filename):
    text_file_path = f"C:/Users/cobyw/OneDrive/Documents/Machine_Learning/Stock_Prices/Total_training_data/{filename}"
    with open(text_file_path, "r") as f:
        print("I actually can read")
        lines = f.readlines()
        day_list = lines[0]
        day_list = eval(day_list)
        f.close()
        return(day_list)
#throw all the stocks randomly into each of the 10 text files. Maybe use a long program?

def run_total_stats(file_list):
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    for n in file_list:
        print("Now reading file", n)
        file_path = f"C:/Users/cobyw/OneDrive/Documents/Machine_Learning/Stock_Prices/{n}.txt"
        with open(file_path, "r") as f:
                lines = f.readlines()
                day_list = lines[0]
                day_list = eval(day_list)
                for n in day_list:              
                    if n[1][0]==1:
                        positive_count +=1
                    if n[1][1]==1:
                        neutral_count+=1
                    if n[1][2]==1:
                            negative_count+=1
                f.close()
    print("positives:",positive_count)
    print("negatives:",negative_count)
    print("neutrals:",neutral_count)
           
def length_counter(file_list):
    total_length = 0
    for n in file_list:
        file_path = f"C:/Users/cobyw/OneDrive/Documents/Machine_Learning/Stock_Prices/Training_by_stock_Month2/{n}.txt"
        with open(file_path, "r") as f:
            print(f"file {n} opened")
            lines = f.readlines()
            day_list = lines[0]
            day_list = eval(day_list)
            length = len(day_list)
            total_length = total_length + length
            print(total_length)
            f.close()
    print("total:",total_length)

def big_document(stocklist,save_name):
    total_list = []
    for stock in stocklist:
        print(f"Thinking about {stock}")
        file_path = f"C:/Users/cobyw/OneDrive/Documents/Machine_Learning/Stock_Prices/Test_data_points/{stock}data.txt"
        with open(file_path, "r") as f:
            lines = f.readlines()
            day_list = lines[0]
            day_list = eval(day_list)
            f.close()  
        total_list = total_list +day_list
        print(f"{stock} added")
    random.shuffle(total_list)
    file_path = f"C:/Users/cobyw/OneDrive/Documents/Machine_Learning/Stock_Prices/{save_name}"
    with open(file_path, "w") as f:
        print("Now writing huge list")
        f.write(f"{total_list}")
        f.close

run_total_stats(["Test_data_prime"])
##11 and 23 are evaluation

#"training data length = 62601" each one is 5691

#Have to throw all the points into one gigantic list

#number of data points = 62,601 data points (month 1)

#update_days(todays_date,stock_symbols, start_date = '2023-01-03', end_date = '2023-01-04')

#For all training data:
#positives: 8678
#negatives : 7241
#neutrals: 103592

#For all evaluation data:
#positives: 840
#negatives: 688
# neutrals: 9854


## I think i need to completely re-evaluate my approach. I need to minimize type 1 error. 
##Keep stochastic gradient descent. Minimizing the cost function can also go negative. If it guesses zero, its technically correct
#However, if it guesses 1 or -1 correctly decrease the cost function by more. But the cost function is the difference, and the difference should be zero.
#The problem is the cost function measures incorrectness not correctness.

#what if i redid all the data to get rid of selling points. 
#conditions:
# If you guess neutral you're fine
#If you guess buy correctly you're great
#If you guess buy incorrectly its very bad
# So what if the neutral guess is just implied
#Time to learn pytorch



#For month 1: 
#positives: 3900
#negatives: 3562
#neutrals: 49448

#For month 2:
#positives: 5275
#negatives: 4071
#neutrals: 59772

#For test data (training_list99)
#positives: 655
#negatives: 288
#neutrals: 9785