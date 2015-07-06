# -*- coding: utf-8 -*-
"""
Code to run the algorithm for the entire dataset with a pre specified window.

@author: t6labs
"""
import numpy as np
import pandas as pd
import algorithm
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FuncFormatter

#Initial portfolio balance: (double)
initCapital = 1000
#Window size used by the anticor algorithm: (integer) > 1
w = 6
#File format should have dates in first column with the header 'Date' and price levels to the right with the ticker symbol as the header
input_file = '/home/user/SPDR.csv'
#text file of the historical weights calculated by day, note the first 2 * w days will be uniform weights
output_weights = '/home/user/histweights.txt'
#Variable used for x-axis spacing on chart
step_interval = 252
#Custom function so that chart will display currency symbols
def currencyfmt(x,pos):
    return '${:0,d}'.format(int(x))
#Data preprocessing from csv
data = pd.read_csv(input_file,sep=',')
#Save the date values for printing on the chart later
x_ticks = data['Date'].values
x_ticks = x_ticks[w * 2:]
data = data.drop('Date',1)
v = data.values
x = v[1:]/v[:-1]
process = algorithm.Algo()
#Initialize portfolio weights and historical weights to uniform
s = x.shape
m = s[1]
b = np.ones(m) * 1.0 / m
hist_weights = b.T
#The primary control statement for the anticor algorithm, for each day obtain the weights and save them in our hist_weights matrix
for t in range(s[0] - 1):
    b = process.anticor(w,t,x,b,False)
    hist_weights = np.vstack((hist_weights,b))
#Output the historical weights to the filename specified at the top        
np.savetxt(output_weights, hist_weights, delimiter=',')
#Calculate the portfolio return by multiply the history of price relatives by the weights    
portfolio = np.sum(np.multiply(hist_weights,x),axis=1)
#Remove the first w*2 days as the anticor algorithm was not active during this time
portfolio = portfolio[w * 2:]
#The cumulative product is the total return over the whole time frame
tot_ret = initCapital * np.cumprod(portfolio)

# Graph the results using matplotlib

formatter = FuncFormatter(currencyfmt)
matplotlib.rcParams.update({'font.size':14})
fig, ax = plt.subplots()
#Sets the y-axis to display as currency
ax.yaxis.set_major_formatter(formatter)
ax.set_ylabel('PnL')
ax.set_title('Anticor Performance')
#Sets the x-axis as dates and spaces them out so you can see them
plt.xticks(np.arange(0,x_ticks.shape[0],step_interval),x_ticks[0::step_interval],rotation=45)
plt.plot(tot_ret)
print tot_ret[-1:]
