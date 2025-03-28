##I need to create some data, a two column pandas table, then send it to excel
##good way to create a datafram is from a list of lists
##Possibly turn tuples into a list of lists
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
coords = [(1,2),(3,6),(5,10),(7,14),(10,20),(13,26)]
xcoords = []
ycoords = []
for n in coords:
    xcoords.append(n[0])
    ycoords.append(n[1])
#For fun, organize in ascending order
#Ok I guess I don't know how to do that

df = pd.DataFrame(coords)
#df.to_csv('test17.csv', sep=',', index=False, encoding='utf-8')

plt.plot(xcoords,ycoords, marker = 'o')
plt.show()


print(df)