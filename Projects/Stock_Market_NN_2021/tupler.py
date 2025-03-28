import random
from stock_storage import stockdict
datalist=[]
crazylist=[]
tuple_list=[]
test_data=[]
training_data=[]
for key in stockdict:
    for x in range(600):
        #guarunteed to get 300 unique points by random selection
        if len(crazylist)!=300: #I need 300 points
            r=random.randint(0,1450) #choose from 1 to 1400 because it can't start any past 1450 if each point is 300 long
            if r not in crazylist: # make sure theyre unique
                crazylist.append(r) #add to the starting point list
#we now have our starting points for one stock
#   
    days=stockdict[key] #This returns all the days in one stock
    for n in crazylist: #i HAVE 500 POINTS
        point=days[n:n+150] #each point is 200 long
        datalist.append(point)
    crazylist.clear()


for j in datalist:
    y=j.pop(0)
    if y>0.5:
        y=1
    else: y=0
    day_tuple=(j,y)
    tuple_list.append(day_tuple)

print(len(tuple_list))  

for b in tuple_list:
    if tuple_list.index(b)%500==0:
        print(tuple_list.index(b))

    if tuple_list.index(b)%5==0:
        test_data.append(b)
    else:         
        training_data.append(b)


text_file = open('test_data4.txt', 'wt')
text_file.write(str(test_data))
text_file.close()

text_file2= open('training_data2.txt','wt')
text_file2.write(str(training_data))
text_file2.close()
