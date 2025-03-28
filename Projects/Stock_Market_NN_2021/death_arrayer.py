
def get_data():
    import random
    from stock_storage import stockdict
    import numpy as np

    datalist=[]
    crazylist=[]
    tuple_list=[]
    test_data=[]
    validation_data=[]
    training_data=[]
    for key in stockdict:
        for x in range(600):
            #guarunteed to get 300 unique points by random selection
            if len(crazylist)!=300: #I need 300 points
                r=random.randint(0,1450) #choose from 1 to 1400 because it can't start any past 1450 if each point is 300 long
                if r not in crazylist: # make sure theyre unique
                    crazylist.append(r) #add to the starting point list
   
       
        days=stockdict[key] #This returns all the days in one stock
        for n in crazylist: #i HAVE 500 POINTS
            point=days[n:n+150] #each point is 200 long
            datalist.append(point)
        crazylist.clear()


    for j in datalist:
        if datalist.index(j)%500==0:
            print(datalist.index(j))
        if datalist.index(j)%5==0:
            y=j.pop(0)
            if y>0.5:
                y=1
            else: y=0   

            p=len(j)
            answer_array=np.array([y])
            between_array=np.array(j)
            question_array = between_array.reshape((p,1))
            day_tuple=(question_array,answer_array)
            test_data.append(day_tuple)

        else:
            h=j.pop(0)
            if h>0.5:
                h=1
            else:
                h=0

            t=len(j)
            answer_array2=np.array(h)
            between_array2=np.array(j)
            question_array2=between_array2.reshape((t,1))
            day_tuple2=(question_array2,answer_array2)
            training_data.append(day_tuple2)
    return(training_data,validation_data,test_data)
print()
    







