import yfinance as yf
import numpy as np
import pandas as pd
import alpaca_trade_api as tradeapi
from alpaca_trade_api.stream import Stream
import asyncio
import os
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import random
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time
from joblib import dump, load
ALPACA_API_KEY = "PKQJ6KTNR6P7AF3Q9UIO"
ALPACA_SECRET_KEY =  "rhe25xMb4LQ1XF07rPxc0NHobbiN9WCBbZzub5LF"
BASE_URL = "https://paper-api.alpaca.markets"
from datetime import datetime, timedelta

aapl_model = load("AAPL_prime.joblib") 
amzn_model =load("AMZN_prime.joblib")
googl_model = load('GOOGL_prime.joblib')
meta_model = load('META_prime.joblib')
msft_model = load('MSFT_prime.joblib')

api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, BASE_URL, api_version='v2')

def calculate_RSI(data,column_title,window=14):
    back1 = data[f'{column_title}'].diff(1) #Another fire function. Calculates the difference between The data and the previous point in the data
    #Remember that you dont actually have to add a column to the dataframe, you can just store it in a variable.
    gain = np.where(back1 > 0, back1, 0).flatten() #if statement, what to append if true, what to append if false
    loss = np.where(back1 < 0, -back1, 0).flatten()
    #This is a cool function stolen from SQL. Creates an array from when a condition in a larger array are met. 
    avg_gain = pd.Series(gain, index=data.index).rolling(window=window, min_periods=1).mean()
    avg_loss =pd.Series(loss,index = data.index).rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


#I now no longer need yfinance to pull historical data. Alpaca has me covered I believe.

def pull_minute_data(ticker,date):
    bars = api.get_bars(ticker, timeframe="1Min", start=date, end=date).df
    bars = bars.reset_index()
    del bars['high'],bars['low'],bars['trade_count'],bars['open'],bars['vwap']
    start_cutoff = f'{date}'+ ' 09:3'
    end_cutoff = f'{date}'+ ' 15:5'
    start_index = bars[bars['timestamp'].astype(str).str.contains(start_cutoff,case=False,na=False)].index[0]
    end_index = bars[bars['timestamp'].astype(str).str.contains(end_cutoff,case=False,na=False)].index[0]
    bars['timestamp'] = bars['timestamp'].dt.tz_localize(None)
    bars2 = bars.iloc[start_index:end_index]
    return bars2

def test_threshold_up(df,row,column):
    rows_belows = df.iloc[row+1:][column]
    threshold_val = df.iloc[row]['threshold_up']
    if ((rows_belows > threshold_val).any()):return 2
    else: return 0


def test_threshold_down(df,row,column):
    rows_belows = df.iloc[row+1:][column]
    threshold_val = df.loc[row,'threshold_down']
    if ((rows_belows<threshold_val).any()): return -1
    else: return 0

def microscopic_factors(ticker,date):
    try:
        df = pull_minute_data(ticker,date)
        ma_twenty = df["close"].rolling(window=20).mean()
        std_dev = df["close"].rolling(window=20).std()
        df['RSI_micro']=calculate_RSI(df,'close')
        df["SMA_12_micro"] = df['close'].rolling(window=12).mean()
        df["SMA_26_micro"] = df['close'].rolling(window=26).mean()
        df["EMA_26_micro"] = df['close'].ewm(span=26, adjust=False).mean()
        df["EMA_12_micro"] = df["close"].ewm(span=12, adjust=False).mean()
        df["MACD_micro"] = df["EMA_12_micro"] - df["EMA_26_micro"]
        df["Signal_Line_micro"] = df["MACD_micro"].ewm(span=9, adjust=False).mean()
        df["Upper_Band_micro"] = ma_twenty + (2 * std_dev)
        df["Lower_Band_micro"] = ma_twenty - (2 * std_dev)
        df['threshold_up'] = df['close']*1.015
        df['threshold_down']=df['close']*.985
        df=df.reset_index()
        positives= df.index.to_series().apply(lambda i: test_threshold_up(df, i, 'close'))
        negatives = df.index.to_series().apply(lambda i: test_threshold_down(df, i, 'close'))
        df['Target1'] = positives + negatives
        df["Target"] = df["Target1"].copy()
        df.loc[df["Target1"] == 2, "Target"] = 1
        df['Date2'] = df['timestamp'].dt.strftime('%Y-%m-%d')
        df.dropna(inplace=True)
        return df
    except: pass


def create_day_range(ticker,start_date,end_date):
    us_holidays = USFederalHolidayCalendar().holidays(start=start_date, end=end_date)
    trading_days = pd.date_range(start=start_date, end=end_date, freq='B')
    trading_days = trading_days[~trading_days.isin(us_holidays)]
    trading_days=trading_days.strftime('%Y-%m-%d')
    master_frame = microscopic_factors(ticker,trading_days[0])
    for date in trading_days[1:]:
        day_frame = microscopic_factors(ticker,date)
        master_frame = pd.concat([master_frame,day_frame],ignore_index=True)

    return master_frame


#given a date range, we need to reate a new range that starts 50 trading days before the start date
def holy_macros(ticker,start_date,end_date):
    date_value = pd.to_datetime(start_date)
    new_start = date_value-pd.Timedelta(days=82)
    data = yf.download(ticker,start=new_start,end=end_date,interval='1d')
    data.dropna(inplace=True)

    data['RSI_14'] = calculate_RSI(data,'Open')
     #This is a fucking awesome function
    data["Log_Return"] = (data["Open"] / data["Open"].shift(1)).apply(lambda x: np.log(x))
    data['Shifted_Log'] = data['Log_Return']
    data["SMA_10"] = data['Open'].rolling(window=10).mean()
    data["SMA_50"] = data['Open'].rolling(window=50).mean()
    data["EMA_50"] = data['Open'].ewm(span=50, adjust=False).mean()
    data["EMA_10"] = data['Open'].ewm(span=10, adjust=False).mean()
    data["EMA_12"] = data["Open"].ewm(span=12, adjust=False).mean()
    data["EMA_26"] = data["Open"].ewm(span=26, adjust=False).mean()
    data["MACD"] = data["EMA_12"] - data["EMA_26"]
    data["Signal_Line"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["20_day_MA"] = data["Open"].rolling(window=20).mean()
    data["20_day_STD"] = data["Open"].rolling(window=20).std()
    data["Upper_Band"] = data["20_day_MA"] + (2 * data["20_day_STD"])
    data["Lower_Band"] = data["20_day_MA"] - (2 * data["20_day_STD"])
    data=data.reset_index()
    data["Date2"] = data['Date'].dt.strftime('%Y-%m-%d')
    #data.dropna(inplace=True)
   
    return data
    


def get_training_data(ticker,start_date,end_date):
    micros = create_day_range(ticker,start_date,end_date)
    macros = holy_macros(ticker,start_date,end_date)
    micros["Date2_"] = pd.to_datetime(micros["Date2"])
    macros["Date2"] = pd.to_datetime(macros["Date2"])
    if isinstance(macros.index, pd.MultiIndex):
        macros = macros.reset_index(drop=True)

    if isinstance(macros.columns, pd.MultiIndex):
        macros.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in macros.columns]

    
    final_combine = pd.merge(macros, micros, on="Date2_", how="outer")
    
    
    final_combine.dropna(inplace=True)
    final_combine['Closing_return'] = (final_combine[f'Close_{ticker}']-final_combine['close'])/final_combine['close']
    final_combine['times'] = final_combine['timestamp'].dt.time
    final_combine['time_fraction'] = (final_combine['times'].astype(str).apply(lambda x: (int(x[0:2])*60)+int(x[3:5]))-570)/390
    return final_combine.copy()




def train_model(ticker,start_date,end_date,test_percent,load=False,save=False,analytics=True):
    if load:
        data = pd.read_pickle(f"{ticker}.pkl").set_index("index")
        print('File Read')
        
        
    else:
        data = get_training_data(ticker,start_date,end_date)
       
        data.to_pickle(f"{ticker}.pkl")
        
        print('File Written')
   
    us_holidays = USFederalHolidayCalendar().holidays(start=start_date, end=end_date)
    trading_days = pd.date_range(start=start_date, end=end_date, freq='B')
    trading_days = trading_days[~trading_days.isin(us_holidays)]
    trading_days=trading_days.strftime('%Y-%m-%d')
    

    features = ['close','volume','RSI_micro','SMA_12_micro','SMA_26_micro','EMA_12_micro','EMA_26_micro',
                'MACD_micro','Signal_Line_micro','Upper_Band_micro','Lower_Band_micro','RSI_14_','Shifted_Log_','SMA_10_',
                'SMA_50_','EMA_10_','EMA_50_','MACD_','Signal_Line_','Upper_Band_','Lower_Band_','time_fraction']
    
    
    acc = []
    returns = []
    for s in range(10):
        test_days = []
        training_days = []
        num_days = len(trading_days)
        test_size = int(num_days*test_percent)
        print('Number of Test Days: ',test_size)
        sample=random.sample(range(len(trading_days)+1),test_size)
        trading_days = pd.to_datetime(trading_days)
        for n in range(len(trading_days)):
            if n in sample:
                
                test_days.append(trading_days[n])
            else:
                
                training_days.append(trading_days[n])

        X_test1 = data[data['Date_'].isin(test_days)]


        X_train1 = data[data['Date_'].isin(training_days)]

        X_test = X_test1[features]
        X_test = X_test.reset_index(drop=True)
        X_train = X_train1[features]
        X_train = X_train.reset_index(drop=True)
        y_train = X_train1['Target']
        y_test = X_test1['Target']
        y_test_return = X_test1['Closing_return']

        d = random.randint(1,100)
        model = RandomForestClassifier(n_estimators=40,random_state=d)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        re = pd.DataFrame(y_test)
        re['Closing_return'] = y_test_return
        re['y_pred'] = y_pred

        conditions = [(re['Target']==re['y_pred'])&(re['Target']==1),
                      (re['Target']==re['y_pred'])&(re['Target']==-1),
                      (re['y_pred']==0),
                      (re['Target']==0)&(re['y_pred']==1),
                      (re['Target']==0)&(re['y_pred']==-1)]
        
        outcomes = .015,.015,0,re['Closing_return'],-1*re['Closing_return']
        re['returns'] = np.select(conditions,outcomes,default=np.nan)
        return_total = re['returns'].sum()
     
        score = model.score(X_test,y_test)
        print(confusion_matrix(y_test, y_pred))
        print(s)
        print('Return: ',return_total)
        print('Accuracy',score)
        returns.append(return_total)
        acc.append(score) 

        if save:
            dump(model, f'{ticker}{s}.joblib')
            print('model updated:',return_total,acc)
    if analytics:
        vals=(np.array(acc))*100
        vals2 = np.array(returns)

        print("Average Accuracy: ",vals.mean())
        print("Average Return: ",vals2.mean())
        print("Minimum Accuracy: ",vals.min())
        print("Minimum Return: ",vals2.min())
        print('Maximum Accuracy: ',vals.max())
        print('Maximum Return: ',vals2.max())
        plt.plot(vals2)
        plt.plot(vals)
        plt.show()

                                        #CURRENT DATA BELOW#
###################################################################################################


def current_micros(ticker,master):
    df=grab_current(ticker,master)
   
    df1 = df.copy()
    if len(df)>27:
       
        ma_twenty = df["close"].rolling(window=20).mean()
        std_dev = df["close"].rolling(window=20).std()
        df['RSI_micro']=calculate_RSI(df,'close')
        df["SMA_12_micro"] = df['close'].rolling(window=12).mean()
        df["SMA_26_micro"] = df['close'].rolling(window=26).mean()
        df["EMA_26_micro"] = df['close'].ewm(span=26, adjust=False).mean()
        df["EMA_12_micro"] = df["close"].ewm(span=12, adjust=False).mean()
        df["MACD_micro"] = df["EMA_12_micro"] - df["EMA_26_micro"]
        df["Signal_Line_micro"] = df["MACD_micro"].ewm(span=9, adjust=False).mean()
        df["Upper_Band_micro"] = ma_twenty + (2 * std_dev)
        df["Lower_Band_micro"] = ma_twenty - (2 * std_dev)
        df['Date2'] = df['timestamp'].dt.strftime('%Y-%m-%d')
        #df.dropna(inplace=True)
           
        return df1,df
    else: return df1,None


def grab_current(ticker,master):
    barset = api.get_latest_bar(ticker)
    df=pd.DataFrame([barset._raw])
    
    df['close'] = df['c']
    df['volume'] = df['v']
    df['timestamp1'] = df['t']
    del df['c'],df['v'],df['o'],df['vw'],df['t'],df['h'],df['l'],df['n']
    df['timestamp1'] = pd.to_datetime(df['timestamp1'])
    df['timestamp'] = df['timestamp1'].dt.tz_convert('America/New_York')
    del df['timestamp1']
    df['timestamp'] = df['timestamp'].dt.tz_localize(None)

    master = pd.concat([master, df], ignore_index=True)
    return master


def current_macros(ticker):##################GOOD
    today = datetime.today().date()
    yesterday = today- timedelta(days=1)
    today=today.strftime('%Y-%m-%d')
    yesterday=yesterday.strftime('%Y-%m-%d')
    date_value = pd.to_datetime(today)
    new_start = date_value-pd.Timedelta(days=82)
    stock = yf.Ticker(ticker)
    recents = stock.history(period="2d")
    yesterday_open=recents.iloc[0]["Open"]
    today_open = recents.iloc[1]["Open"]

    data = yf.download(ticker,start=new_start,end=today,interval='1d')
    if isinstance(data.index, pd.MultiIndex):
        data = data.reset_index(drop=True)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in data.columns]
    
    data.dropna(inplace=True)
    del data[f'Close_{ticker}'],data[f'High_{ticker}'],data[f'Low_{ticker}'],data[f'Volume_{ticker}']
    data['Open'] = data[f'Open_{ticker}']
    del data[f'Open_{ticker}']
    data.reset_index(inplace=True)
    data.loc[len(data),'Open']=yesterday_open
    data.loc[len(data)-1,'Date']=yesterday
    data.loc[len(data),'Open']=today_open
    data.loc[len(data)-1,'Date']=today
    data = data.drop_duplicates()
    #add the new opening values to the bottom of open
    data['RSI_14'] = calculate_RSI(data,'Open')
     #This is a fucking awesome function
    data["Log_Return"] = (data["Open"] / data["Open"].shift(1)).apply(lambda x: np.log(x))
    data['Shifted_Log'] = data['Log_Return']
    data["SMA_10"] = data['Open'].rolling(window=10).mean()
    data["SMA_50"] = data['Open'].rolling(window=50).mean()
    data["EMA_50"] = data['Open'].ewm(span=50, adjust=False).mean()
    data["EMA_10"] = data['Open'].ewm(span=10, adjust=False).mean()
    data["EMA_12"] = data["Open"].ewm(span=12, adjust=False).mean()
    data["EMA_26"] = data["Open"].ewm(span=26, adjust=False).mean()
    data["MACD"] = data["EMA_12"] - data["EMA_26"]
    data["Signal_Line"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["20_day_MA"] = data["Open"].rolling(window=20).mean()
    data["20_day_STD"] = data["Open"].rolling(window=20).std()
    data["Upper_Band"] = data["20_day_MA"] + (2 * data["20_day_STD"])
    data["Lower_Band"] = data["20_day_MA"] - (2 * data["20_day_STD"])
    data=data.reset_index(drop=True)
    data["Date2"] = data['Date'].dt.strftime('%Y-%m-%d')
    data = data.rename(columns=lambda x: x + '_')
    data.dropna(inplace=True)
    
    return data


def add_macros(ticker,micros,macros):
    micros["Date2_"] = pd.to_datetime(micros["Date2"])
    macros["Date2_"] = pd.to_datetime(macros["Date2_"])

    
    final_combine = pd.merge(macros, micros, on="Date2_", how="outer")
    final_combine.dropna(inplace=True)
    final_combine['times'] = final_combine['timestamp'].dt.time
    final_combine['time_fraction'] = (final_combine['times'].astype(str).apply(lambda x: (int(x[0:2])*60)+int(x[3:5]))-570)/390
    return final_combine.copy()


def do_i_buy(output,amount,ticker,account_balance,price):
    if account_balance>amount:

        if output ==1:
            try:
                buy_order = api.submit_order(
                symbol=ticker,
                notional=round(amount,2),
                side='buy',
                type='market',
                time_in_force='day')  
                print(f"Buy order placed for {ticker} at {price}")

                print("Made it here")
                take_profit_order = api.submit_order(
                    symbol=ticker,
                    notional=round(amount,2),
                    side='sell',
                    type='limit',
                    time_in_force='day',
                    limit_price=round((price*1.015),2)
                    )
                print('Made it here')
                stop_loss_order = api.submit_order(
                    symbol=ticker,
                    notional=round(amount,2),
                    side='sell',
                    type='stop',
                    time_in_force='day',
                    stop_price=round((price*.98),2)
                    )
            except: print(f'No {ticker} order submitted: Hold')


        if output ==-1:
            try:
                buy_order = api.submit_order(
                symbol=ticker,
                notional=round(amount,2),
                side='sell',
                type='market',
                time_in_force='day'
                )
                print(f"Sell order placed for {ticker} at {price}")

                take_profit_order = api.submit_order(
                    symbol=ticker,
                    notional=round(amount,2),
                    side='buy',
                    type='stop',
                    time_in_force='day',
                    limit_price=round((price*1.015),2)
                )

                stop_loss_order = api.submit_order(
                    symbol=ticker,
                    notional=round(amount,2),
                    side='buy',
                    type='limit',
                    time_in_force='day',
                    stop_price=round((price*.98),2)
                )
            except: print(f'No {ticker} order submitted: Hold')


        if output==0:
            print(f'No {ticker} order submitted: Hold')


def monitor_live(ticker,master_frame,trade_amount,bankroll,macros,i,model):


    features = ['close','volume','RSI_micro','SMA_12_micro','SMA_26_micro','EMA_12_micro','EMA_26_micro',
                'MACD_micro','Signal_Line_micro','Upper_Band_micro','Lower_Band_micro','RSI_14_','Shifted_Log_','SMA_10_',
                'SMA_50_','EMA_10_','EMA_50_','MACD_','Signal_Line_','Upper_Band_','Lower_Band_','time_fraction']


    print(f"Checking Data for {ticker}, timepoint ",i)
    data = current_micros(ticker,master_frame)
    unmicroed=data[0]
    microed = data[1]
   
    if len(unmicroed)>27:
        prediction=add_macros(ticker,microed,macros)
        point = prediction.iloc[-1]
        
        point2 = point[features]
        
        point3 = pd.DataFrame(point2)
        decision = model.predict(point3.T)
        integer = decision[0]
        latest_trade = api.get_latest_trade(ticker)
        current_price = latest_trade.price
        do_i_buy(integer,trade_amount,ticker,bankroll,current_price)
    return data[0]


def start_import():
    account = api.get_account()
    starting_balance = float(account.cash)

    trade_amount = starting_balance/3000
    aapl = pd.DataFrame()
    amzn = pd.DataFrame()
    meta = pd.DataFrame()
    googl = pd.DataFrame()
    msft = pd.DataFrame()

    macros_aapl = current_macros('AAPL')
    macros_amzn = current_macros('AMZN')
    macros_meta = current_macros('META')
    macros_googl = current_macros('GOOGL')
    macros_msft = current_macros('MSFT')
    print("Macros Created")
    i=0
    while True:
        i+=1
        closing_time = datetime.strptime("16:55", "%H:%M").time()
        current_time = datetime.now().time()

        if current_time < closing_time:
            balance = float(account.cash)

            aapl=monitor_live('AAPL',aapl,trade_amount,balance,macros_aapl,i,aapl_model)
            balance = float(account.cash)
            amzn=monitor_live('AMZN',amzn,trade_amount,balance,macros_amzn,i,amzn_model)
            balance = float(account.cash)
            googl=monitor_live('GOOGL',googl,trade_amount,balance,macros_googl,i,googl_model)
            balance = float(account.cash)
            msft=monitor_live('MSFT',msft,trade_amount,balance,macros_msft,i,msft_model)
            balance = float(account.cash)
            meta=monitor_live('META',meta,trade_amount,balance,macros_meta,i,meta_model)
            time.sleep(40)
        else:
            close_up_shop()
            final_balance = float(account.cash)
            day_profit = final_balance-starting_balance
            day_return = day_profit/starting_balance
            print("Today's Profit: ",day_return)
            print("Today's Return: ",day_return)
            False
            

def close_up_shop():
    positions = api.list_positions()
    for position in positions:
        symbol = position.symbol
        qty2 = float(position.qty)  
        qty1 = round(qty2,2)

        if qty1 > 0:
            print(f"Selling {qty1} shares of {symbol} (long position).")
            api.submit_order(symbol=symbol,qty=qty1,side='sell',type='market',time_in_force='day')

        elif qty1 < 0:
            print(f"Buying back {abs(qty1)} shares of {symbol} (short position).")
            api.submit_order(symbol=symbol,qty=abs(qty1),side='buy',type='market',time_in_force='day')

#train_model('META','2022-08-11','2025-01-10',.15,load=False,save=True,analytics=True)
            
#train_model('META','2022-10-11','2025-03-19',.15,load=False,save=True,analytics=True)

#My last job is converting my output into a trainable point. Lets check the order of the 


start_import()
