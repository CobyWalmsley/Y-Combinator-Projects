import yfinance as yf
import pandas as pd
import datetime 
from matplotlib import pyplot as plt
import numpy as np
default = 'DOGZ'
default_range = '1m'


def pull_numeric(ticker,param):
    stock=yf.Ticker(ticker)
    value = stock.info.get(param)
    print(value)
    value=str(value)
    if value==None:
        return(None)
    elif value != None:
        if len(value)==14:
            out = (value[0:1]+"."+value[2:4]+' Trillion')
            return out
        elif len(value)==13:
            out = (value[0]+"."+value[1:3]+' Trillion')
            return out
        elif len(value)==12:
            out = (value[0:2]+"."+value[3:4]+' Billion')
            return out
        elif len(value)==11:
            out = (value[0:1]+"."+value[2:3]+' Billion')
            return out
        elif len(value)==10:
            out = (value[0]+"."+value[1:2]+' Billion')
            return out
        elif len(value)==9:
            out = value[0:2]+'.'+value[3:4]+'Million'
        elif len(value)==8:
            out = value[0:1]+'.'+value[2:3]+' Million'
        elif len(value)==7:
            out = value[0]+'.'+value[1:2]+' Million'
        elif len(value)<7:
            return value

#print(pull_numeric("AAPL","trailingEps"))
ticker = "ABCL"
stock = yf.Ticker(ticker)

# Print all available info
print(stock.info.get('longName'))