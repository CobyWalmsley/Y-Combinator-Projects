import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=yf.download('AMZN',start='2024-01-10', interval='1m',  end='2024-01-11',progress=False)[['Close']]
data.head()
 
string_data=data['Close'].tolist()
day_array=np.array(string_data)

x_axis_list=[]
for x in range(0,len(string_data)):
    x_axis_list.append(x)

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
no_cut_profit=0
max_profit=0
last_sell=0

for n in range(12,385):
        
        if no_cut_profit>10:
            emergency_cut=0

        if profit>max_profit:
            max_profit=profit

        if binary_gains==3:
            emergency_cut=1
             

        current_gain=0
        current_price=string_data[n]
        
        if condition==1:

            current_gain=current_price-original_buy
            if current_gain>max_gain:
                max_gain=max_gain+(current_gain-max_gain)

        sell_check=0
        
        gain_adjusted=0.90*max_gain
        if gain_adjusted<current_gain:
            sell_check=1

        slope_list=[]
        for t in range(3,15):
            slope=trendline(x_axis_list,string_data,t,n)
            if slope>0: slope=1
            if slope<=0: slope=0
            slope_list.append(slope)
            sliced_list=slope_list[0:4]


            if t==14:
                    

                    if condition==0:
                        

                        if sliced_list==[1,1,1,1]:
                            buy_list.append(n)
                            
                            original_buy=current_price
                            condition=1
                            max_gain=0
                            current_gain=0
                            if emergency_cut==0:
                                plt.axvline(x=n,color='green')
                            
                            

                        if sliced_list==[1,1,1,0]:
                            buy_list.append(n)
                            if emergency_cut==0:
                                plt.axvline(x=n,color='green')
                            original_buy=current_price
                            condition=1
                            max_gain=0
                            current_gain=0
                            
                            
                        
                        if sliced_list==[1,1,0,0]:
                            buy_list.append(n)
                            if emergency_cut==0:
                                plt.axvline(x=n,color='green')
                            original_buy=current_price
                            condition=1
                            max_gain=0
                            current_gain=0
                        
                        if current_price>1.003*last_sell:
                            buy_list.append(n)
                            if emergency_cut==0:
                                plt.axvline(x=n,color='green')
                            original_buy=current_price
                            condition=1
                            max_gain=0
                            current_gain=0
                                

                    if condition==1:
                        if emergency_cut==0:
                            if n>20:
                                if no_cut_profit+current_gain<-15:
                                    emergency_cut=1
                                    condition=0
                                    sell_list.append(n)
                                    plt.axvline(x=n,color='red')
                                    profit=profit+(current_gain)
                                    last_sell=current_price
                                    
                                    no_cut_profit=no_cut_profit+current_gain
                                    if current_gain>0: 
                                        current_gain=1
                                        binary_gains=0
                                    if current_gain<=0: 
                                        current_gain=0
                                        binary_gains+=1
                                    


                        if current_price<(.996)*(max_gain+original_buy):
                            sell_list.append(n)
                            if emergency_cut==0:
                                plt.axvline(x=n,color='red')
                                profit=profit+(current_gain)
                            condition=0
                            no_cut_profit=no_cut_profit+current_gain
                            if current_gain>0: 
                                current_gain=1
                                binary_gains=0
                            if current_gain<=0: 
                                current_gain=0
                                binary_gains+=1
                            last_sell=current_price
                            

                        if max_gain>original_buy*1.01:
                            if current_gain<(.95)*max_gain:
                                sell_list.append(n)
                                if emergency_cut==0:
                                    plt.axvline(x=n,color='red')
                                    profit=profit+(current_gain)
                                condition=0
                                no_cut_profit=no_cut_profit+current_gain
                                if current_gain>0: 
                                    current_gain=1
                                    binary_gains=0
                                if current_gain<=0: 
                                    current_gain=0
                                    binary_gains+=1
                                last_sell=current_price
                                

                        if sell_check==1:

                            if sliced_list==[0,0,1,1]:

                                    sell_list.append(n)
                                    if emergency_cut==0:
                                        plt.axvline(x=n,color='red')
                                        profit=profit+(current_gain)
                                    condition=0
                                    no_cut_profit=no_cut_profit+current_gain
                                    if current_gain>0: 
                                        current_gain=1
                                        binary_gains=0
                                        
                                    if current_gain<=0: 
                                        current_gain=0
                                        binary_gains+=1
                                    last_sell=current_price
                                   

                            if sliced_list==[0,0,1,1]:
                                
                                    sell_list.append(n)
                                    if emergency_cut==0:
                                        plt.axvline(x=n,color='red')
                                        profit=profit+(current_gain)
                                    condition=0
                                    no_cut_profit=no_cut_profit+current_gain
                                    if current_gain>0: 
                                        current_gain=1
                                        binary_gains=0
                                    if current_gain<=0: 
                                        current_gain=0
                                        binary_gains+=1
                                    last_sell=current_price
                                   
                                    
                            if sliced_list==[1,0,0,0]:
                                
                                    sell_list.append(n)
                                    if emergency_cut==0:
                                        profit=profit+current_gain
                                        plt.axvline(x=n,color='red')
                                    condition=0
                                    no_cut_profit=no_cut_profit+current_gain
                                    if current_gain>0: 
                                        current_gain=1
                                        binary_gains=0
                                    if current_gain<=0: 
                                        current_gain=0
                                        binary_gains+=1
                                    last_sell=current_price
                                   
                            

                        if n==384:  
                            sell_list.append(n)
                            condition=0
                            if emergency_cut==0:        
                                plt.axvline(x=n,color='red')
                                profit=(profit+current_gain)

profit_without=string_data[384]-string_data[1]      

print('Buy List',buy_list)
print('sell_list',sell_list)
print(profit,'profit')
print(profit_without, 'profit without')

print(string_data[1],string_data[384])

plt.show()
