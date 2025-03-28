import pandas as pd
import yfinance as yf
import numpy as np
import datetime as dt
import time
import pandas_datareader.data as web

df=pd.read_excel("C:/Users/cobyw/OneDrive/Documents/Voyage Analytics/MFE/Biotech/2024 Releases.xlsx")
print(df)
frame = []

for d in df.columns:
    if '.' in d:
        n = d[:-2]
    else: 
        n=d
    prices=[]
    stock=yf.Ticker(n)
    for y in df[d]:
        try:
            price = stock.history(start=y, end=y + pd.Timedelta(days=1))
            prices.append(price['Open'].iloc[0])
            print(n,y,price['Open'].iloc[0])
            time.sleep(.2)
        except:
            print("nah")
            prices.append("Null")
    frame.append(prices)
df2 = pd.DataFrame(frame)
df2.to_excel('2024_Openings1.xlsx', index=False)      

#making decisions is a funny thing
