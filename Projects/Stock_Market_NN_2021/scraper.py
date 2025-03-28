from decimal import Decimal
from selenium import webdriver
import time
from datetime import datetime
import csv
PATH = 'C:/Users/cobyw/Documents/College Textbooks/Important Documents/chromedriver.exe'
driver = webdriver.Chrome(PATH)
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
driver.get(url)

sp500=[]
sp500.clear()
for x in range(1,506): #506 how many stocks

    current_path = f'/html/body/div[3]/div[3]/div[5]/div[1]/table[1]/tbody/tr[{x}]/td[1]'

    title = driver.find_element_by_xpath(current_path)
    text=title.text
    print(x,text)
    sp500.append(text)

stockdict={}
stockdict.clear()

for symbol in sp500:
    stock_url= f'https://finance.yahoo.com/quote/{symbol}/history?period1=1422144000&period2=1643068800&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true'
    driver.get(stock_url)
#Everything up to here is perfect

    float_list=[]
    res = []
    final=[]
    last_height=657

    for k in range(1,75):
            last_height=last_height+1000
            driver.execute_script(f"window.scrollTo(0, {last_height});")
            time.sleep(.01)

    for j in range(1,1764): #1764 how many days
        if j%100==0:
            print(j)
        price_path=f'/html/body/div[1]/div/div/div[1]/div/div[3]/div[1]/div/div[2]/div/div/section/div[2]/table/tbody/tr[{j}]/td[2]/span'
        digit = driver.find_element_by_xpath(price_path)
        digit_text=digit.text
        float_list.append(digit_text)

    for c in float_list:
        if c!='Dividend':
            if c!= 'Stock Split':
                final.append(c)

    for m in final:
        m=m.replace(',','')
        value=float(m)
        res.append(value)
    print(len(res))    
       
        #everything up to here works
    product_list=[]
    for n in range(0,1725): #how many days-(total/500)*12
        datasum=res[n+1]-res[n]
        data_divisor=datasum/res[n]
        round_data2=round(data_divisor,6)
        neuron_data=(0.5+round_data2)
        rounded_neuron=round(neuron_data,5)
        product_list.append(neuron_data)
    

    if j==1763: #how many days-1 
        stockdict[symbol]=product_list

text_file = open('full_stock_text.txt', 'wt')
text_file.write(str(stockdict))
text_file.close()
