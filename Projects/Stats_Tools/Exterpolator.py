import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.dates as mdates
df = pd.read_excel('C:/Users/cobyw/OneDrive/Documents/Voyage Analytics/MFE/Balance.xlsx')
print(df.head())
#df.plot(x='meeting date',y='long rate sentiment')
meeting_date = np.array(df['meeting date'].to_list())


end_num = mdates.date2num(pd.to_datetime('2008-01-02'))
start_num = mdates.date2num(pd.to_datetime('2024-07-31'))
expanded_x = np.linspace(start_num,end_num,4158)
meeting_nums = []
for n in meeting_date:
    meeting_nums.append(mdates.date2num(pd.to_datetime(n)))


def expand_pandas(yvals):
    interp = interp1d(meeting_nums, yvals, kind='linear', fill_value="extrapolate")
    expanded_y = interp(expanded_x)
    return(expanded_y,expanded_x)

def total_expander():
    total_data = []
    for col in df.columns[1:]:
        useful_list = np.array(df[col].to_list())
        outputs = expand_pandas(useful_list)
        total_data.append(outputs[0])
    return(total_data,outputs[1])

def pandas_constructor(meetings):
    data1 = total_expander()
    data = data1[0]
    meetings = data1[1]
    data.append(meetings)
    meeting_dates = []
    for n in meetings:
        meeting_dates.append(mdates.num2date(n))
    data.append(meeting_dates)
    df2 = pd.DataFrame(data).T
    df2.to_csv('Expanded_Balance.csv', index=False)

pandas_constructor(meeting_nums)
    



#time to reconstruct the dataframe with every heading expanded
#plt.plot(expanded_x,expanded_y)
