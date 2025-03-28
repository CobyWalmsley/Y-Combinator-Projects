#make a thing that identifies peaks, valleys, ups and downs, and flatlines and basically oversimplifies the graph so that it can make decisions
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

point_value=295

data=yf.download('MU',start='2022-03-02', interval='1m',  end='2022-03-03',progress=False)[['Close']]
data.head()
 

string_data=data['Close'].tolist()
day_array=np.array(string_data)



x_axis_list=[]
for x in range(0,len(string_data)):
    x_axis_list.append(x)

x_axis_array=np.array(x_axis_list)

# n is how many points back from the current point, point_number is what actual number of point we are on


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


plt.plot(x_axis_array,day_array)

total_slopes={}
buy_list=[]
sell_list=[]
profit=0
max_gain=0
condition=0
original_buy=0
emergency_cut=0
binary_gains=0
override=0
current_gain=0
wake_up_alarm=0
current_max=0
current_min=0

current_min=string_data[0]
current_max=string_data[0]
for n in range(10,385): #385
        

        current_price=string_data[n] #The current price
        current_gain=0 #Current gain must equal zero at the beginning of every minute

        if current_price>=current_max: current_max=current_price
        if current_price<=current_min: current_min=current_price

        drop_length=current_max-current_min
        current_drop=current_max-current_price

        

         #length of the drop is the previous max minus the previous min
        

        if condition==1: #If we're open for business (3  consecutive losses havent happened)
            current_gain=current_price-original_buy  #current_gain is the difference between the current price and the original buy price

            if current_gain>max_gain:
                max_gain=current_gain

        if condition==0:
            if current_price<current_min:
                current_price=current_min

            if current_drop<0.94*drop_length:
                buy_list.append(n) #add to the buy list
                original_buy=current_price # set the next original price
                condition=1 #time to sell
                max_gain=0 #max_gain starts at zero
                current_gain=0 #so does current gain
                current_max=current_price
                current_min=current_price
                plt.axvline(x=n,color='green') #plot the point
                  
                  
                    

        if condition==1:
            if current_price<(original_buy+max_gain)*.996: #lock
                plt.axvline(x=n,color='red')
                profit=(profit+current_gain)
                if current_gain>0: 
                    current_gain=1
                    binary_gains=0
                if current_gain<=0: 
                    current_gain=0
                    binary_gains+=1
                condition=0

            if current_drop<0.05*original_buy:
                if max_gain>.002*original_buy:
                    if current_gain<.6*max_gain:
                        plt.axvline(x=n,color='red')
                        profit=(profit+current_gain)
                        current_drop=0
                        drop_length=0
                        if current_gain>0: 
                            current_gain=1
                            binary_gains=0
                        if current_gain<=0: 
                            current_gain=0
                            binary_gains+=1
                        condition=0
            
                
            if max_gain==0:
                if current_gain<-.0007*original_buy:    
                    plt.axvline(x=n,color='red')
                    profit=(profit+current_gain)
                    current_drop=0
                    drop_length=0
                    if current_gain>0: 
                        current_gain=1
                        binary_gains=0
                    if current_gain<=0: 
                        current_gain=0
                        binary_gains+=1
                    condition=0
                                   
 
        if condition==1:    
            if n==384:  
                sell_list.append(n)
                condition=0           
                plt.axvline(x=n,color='red')
                profit=(profit+current_gain)
print('profit:',profit)
profit_without=string_data[384]-string_data[0]
print('profit without:',profit_without)

  
            
plt.show()
# after selling, initiate a a thing tracking a current minimum, and only buy once the price has risen a certain percentage above that current min
#after buying, do the max gain thing and only sell after a certain percentage of the max gain has been lost.
#The rules are: keep track of the last current maximum and minimum, along with the current gain. If the increase is higher than the difference between the last maximum
# and the current minimum, buy. If you lose 10% of your max gain, sell.

