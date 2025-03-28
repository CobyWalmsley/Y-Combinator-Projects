import random
import numpy as np
import json
import sys
from decimal import Decimal
from selenium import webdriver
import time
from datetime import datetime
import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from sp500 import sp500


username_str='cjw325@lehigh.edu'
password_str='Aviators@13$$'
PATH = r'C:/Users/cobyw/Documents/Important Documents/stock market/chromedriver.exe'
driver=webdriver.Chrome(PATH)
url='https://sso.accounts.dowjones.com/login?state=hKFo2SA5T2JvUXJuZ0M1ajdwTDJ3bUE5TkFfNjVHUXlWLTlQcqFupWxvZ2luo3RpZNkgZnplWTFsSTdSMXNPVnp0SUh6LUh0SjZRZjZHX012UHejY2lk2SA1aHNzRUFkTXkwbUpUSUNuSk52QzlUWEV3M1ZhN2pmTw&client=5hssEAdMy0mJTICnJNvC9TXEw3Va7jfO&protocol=oauth2&scope=openid%20idp_id%20roles%20email%20given_name%20family_name%20djid%20djUsername%20djStatus%20trackid%20tags%20prts%20suuid%20createTimestamp&response_type=code&redirect_uri=https%3A%2F%2Faccounts.marketwatch.com%2Fauth%2Fsso%2Flogin&nonce=7faf11c8-1ba9-413b-b484-72a575f97709&ui_locales=en-us-x-mw-3-8&ns=prod%2Faccounts-mw#!/signins'
driver.get(url)



time.sleep(1)

username = driver.find_element_by_id('username')
username.send_keys(username_str)
time.sleep(1)

confirm_user=driver.find_element_by_class_name('solid-button continue-submit new-design')

confirm_user.click()
confirm_user.click()
confirm_user.click()



