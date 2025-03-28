from calendar import different_locale
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=yf.download('AMZN',start='2022-03-09', interval='1m',  end='2022-03-10',progress=False)[['Close']]
data.head()

string_data=data['Close'].tolist()
trash_value=string_data.pop(-1)
day_array=np.array(string_data)
x_axis_list=[]
for x in range(0,len(string_data)):
    x_axis_list.append(x)
moving_average_list=x_axis_list[4:]
moving_average_x_array=np.array(moving_average_list)

x_axis_array=np.array(x_axis_list)

def trendline(x_original,y_original,n,point_number):
    x_set=x_original[0:point_number]
    y_set=y_original[0:point_number]
    x_points=x_set[-n:]
    y_points=y_set[-n:]
    x_total=0
    for o in x_points:
        x_total=x_total+o
    y_total=0
    for g in y_points:
        y_total=y_total+g
    squared_x=0
    for k in x_points:
        p=k**2
        squared_x=squared_x+p
    total_zipped=0
    for x,y in zip(x_points,y_points):
        z=x*y
        total_zipped=total_zipped+z
    a=total_zipped*len(x_points)
    b=y_total*x_total
    c=squared_x*len(x_points)
    d=(x_total**2)
    t=(a-b)
    f=(c-d)
    m=t/f
    f=m*x_total
    intercept=(y_total-f)/len(x_points)
    
    x=np.linspace((point_number-n),point_number,500)
    y=m*x+intercept
   
    return m

def moving_average(x_points, current_min, points_back):
    difference=current_min-points_back
    calculations=x_points[difference:current_min]
    total=0
    for n in calculations:
        total=total+n
    mean=total/points_back
    calculations=[]
    return mean


plt.plot(x_axis_array,day_array)

buy_list=[]
sell_list=[]

profit=0 #total dollars earned, added to every sell
max_gain=0 #resets every sell, the most gain that has been acheived (but not acted on) between buying and selling
condition=0 # whether the bot has most recently bought or sold. 0= time to buy, 1=time to sell
original_buy=0 # resets every buy, the price at which the bot last bought the stock
emergency_cut=0 # whether or not the emergency cut has kicked in
binary_gains=0 # how many failed sales in a row the bot has made. If three sales in a row that have had losses are made, the emergency cut kicks on.
no_cut_profit=0 # for comparison's sake, the profit that would have been made if the emergency cut safety feature wasn't included
max_profit=0 # similar to max gain, but for profit instead of gain
last_sell=0 # similar to original buy, but for selling
current_gain=0
moving_averages=[]

ideal_buys=[]
ideal_sells=[]
#Buy if the trendlines turn around 
# sell if a certain portion of the max gain is lost, or if the current gain dips negative
one_percent=(string_data[0])/100
print(one_percent,'1%')
for n in range(4,len(string_data)): #check every point from number 10 to 384 in the day, to allow the first few minutes to build up.
    
    average=moving_average(string_data,n,4)
    moving_averages.append(average)
    current_price=string_data[n]
    current_gain=current_price-original_buy
    moving_ave_difference=current_price-average

    factor=one_percent/moving_ave_difference

    if profit>max_profit:
        max_profit=profit

    if n>7:
        moving_average_slope_list=[]
        for t in range(3,9):
            moving_ave_slope=trendline(moving_average_list,moving_averages,t,n)
            if moving_ave_slope>0: moving_ave_slope=1
            if moving_ave_slope<=0: moving_ave_slope=0
            moving_average_slope_list.append(moving_ave_slope)

    last_sell=string_data[0]

    if condition==1:
        if n<10:
            if moving_ave_difference<-5:
                sell_list.append(n)           
                plt.axvline(x=n,color='red')
                profit=(profit+current_gain)
                condition=0


    if n==4:
        if condition==0:
            condition=1
            buy_list.append(n)
            max_gain=0
            current_gain=0
            original_buy=current_price
            if emergency_cut==0:
                plt.axvline(x=n,color='green')
               
    
    if n>8:
        slope_list=[]
        for t in range(3,8):
            slope=trendline(x_axis_list,string_data,t,n)
            if slope>0: slope=1
            if slope<=0: slope=0
            slope_list.append(slope)
            sliced_list=slope_list[0:4]

        if condition==0:
            
            








            
               
        if condition==1:
                current_gain=current_price-original_buy
                
                if n==387:
                    sell_list.append(n)
                    condition=0           
                    if emergency_cut==0:
                        plt.axvline(x=n,color='red')
                        profit=(profit+current_gain)

        
    if n in ideal_buys:
            print(n,'buy')
            print(moving_average_slope_list,'moving ave slope list')
            print(sliced_list,'point slope list')
            print(moving_ave_difference,'difference')
            print(one_percent,'one percent')
            print(factor,'factor')
    if n in ideal_sells:
        print(n,'sell')
        print(moving_average_slope_list,'moving ave slope list')
        print(sliced_list,'point slope list')
        print(moving_ave_difference,'difference')
        print(one_percent,'one percent')
        print(factor,'factor')
        print(current_gain,'current gain')
                
moving_average_y_array=np.array(moving_averages)

profit_without=string_data[-1]-string_data[1]      

print('Buy List',buy_list)
print('sell_list',sell_list)
print(profit,'profit')
print(profit_without, 'profit without')
plt.plot(moving_average_x_array,moving_average_y_array)
print(string_data[1],string_data[-1])
plt.show()
