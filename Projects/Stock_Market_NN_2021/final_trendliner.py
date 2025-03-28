import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

point_value=295

data=yf.download('AMZN',start='2022-08-16', interval='1m',  end='2022-08-17',progress=False)[['Close']]
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
points_to_check=[11,19,22,39,53]
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
for n in range(12,385):

        if binary_gains==3:  #If three losses happen in a row, cut for the day
            emergency_cut=1

           
        
        current_price=string_data[n] #The current price
        current_gain=0 #Current gain must equal zero at the beginning of every minute
        
        if condition==1: #If we're open for business (3  consecutive losses havent happened)
            current_gain=current_price-original_buy  #current_gain is the difference between the current price and the original buy price

            if current_gain>max_gain:
                max_gain=current_gain
            #You're only allowed to sell if the current gain falls below 80% of the max gain, or if the gain goes 5% negative, so no sell permission needed
        if n>50:   
            wake_up_test=trendline(x_axis_list,string_data,40,n)
            if emergency_cut==1:
                if wake_up_test>0:
                    wake_up_alarm+=1
                if wake_up_test<0:
                    wake_up_test=0
                if wake_up_alarm==37:
                    buy_list.append(n) #add to the buy list
                    original_buy=current_price # set the next original price
                    condition=1 #time to sell
                    max_gain=0 #max_gain starts at zero
                    current_gain=0 #so does current gain
                    plt.axvline(x=n,color='green') #plot the point
                  



        if wake_up_alarm<30:

            long_slope_list=[]
            for t in range(3,9): #calculate the last four trendlines back
                slope=trendline(x_axis_list,string_data,t,n)
                if slope>0: slope=1
                if slope<=0: slope=0 #make them binary
                long_slope_list.append(slope)
        
                slope_list=long_slope_list[0:4]

                selling_threshold=0.98*max_gain

                if t==6: # wait til the slope list is full before buying or selling
            

                    if condition==0: #If its buying's turn
                        if slope_list==[1,1,1,1]:
                            buy_list.append(n) #add to the buy list
                            original_buy=current_price # set the next original price
                            condition=1 #time to sell
                            max_gain=0 #max_gain starts at zero
                            current_gain=0 #so does current gain
                            if emergency_cut==0: #If we're open for business
                                plt.axvline(x=n,color='green') #plot the point
                            
                        if slope_list==[1,1,1,0]:
                            buy_list.append(n) #add to the buy list
                            original_buy=current_price # set the next original price
                            condition=1 #time to sell
                            max_gain=0 #max_gain starts at zero
                            current_gain=0 #so does current gain
                            if emergency_cut==0: #If we're open for business
                                plt.axvline(x=n,color='green') #plot the point

                        if slope_list==[1,1,0,0]:
                            buy_list.append(n) #add to the buy list
                            original_buy=current_price # set the next original price
                            condition=1 #time to sell
                            max_gain=0 #max_gain starts at zero
                            current_gain=0 #so does current gain
                            if emergency_cut==0: #If we're open for business
                                plt.axvline(x=n,color='green') #plot the point
                
            
                    cut_losses=0.993*original_buy
                    if condition==1:
                        
                        # You're only allowed to sell if the current gain falls below 80% of the max gain after the gain has increased 5% or if the price goes 5% negative
                        if (max_gain+original_buy)>=1.003*original_buy: #If at any point you've been up .5%:
                            if current_gain<selling_threshold:
                                sell_list.append(n)
                                condition=0
                                
                                if emergency_cut==0:
                                    plt.axvline(x=n,color='red')
                                    profit=(profit+current_gain)
                                if current_gain>0: 
                                    current_gain=1
                                    binary_gains=0
                                if current_gain<=0: 
                                    current_gain=0
                                    binary_gains+=1

                        if cut_losses>current_price:
                            if current_gain<selling_threshold:
                                sell_list.append(n)
                                condition=0
                                
                                if emergency_cut==0:
                                    plt.axvline(x=n,color='red')
                                    profit=(profit+current_gain)
                                if current_gain>0: 
                                    current_gain=1
                                    binary_gains=0
                                if current_gain<=0: 
                                    current_gain=0
                                    binary_gains+=1
        if condition==1:    
            if n==384:  
                sell_list.append(n)
                condition=0           
                plt.axvline(x=n,color='red')
                profit=(profit+current_gain)
            
        
        

profit_without=string_data[384]-string_data[1]
print(profit,'profit')
percent_profit_with=(profit/string_data[1])*100
print(profit_without,'profit without')
percent_profit_without=(profit_without/string_data[1])*100
print('percent with', percent_profit_with)
print('percent without',percent_profit_without)
plt.show()


