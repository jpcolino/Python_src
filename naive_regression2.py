from pandas import Series, DataFrame,isnull, ExcelFile, date_range
from pylab import plot, show
import pandas as pd
import pylab as pl
import numpy as np

# 1. Reading data
path_xls = 'C:\Users\Suso\Desktop\Ipython Notebooks\Naivereg1.csv'
xls_data = pd.read_csv(path_xls,index_col='Date')

# TEST: 

xls_data = pd.read_csv(path_xls,parse_dates=True, index_col=0)
prices = xls_data.dropna()
commdty_corr = lambda x: x.corrwith(x['Power'])
by_year = prices.groupby(lambda x: x.year)
by_year.apply(commdty_corr).plot()
import statsmodels.api as sm
def regress(data,yvar,xvars):
    Y = data[yvar]
    X = data[xvars]
    X['intercept'] = 1.
    result = sm.OLS(Y,X).fit()
    return result.params
    
by_year.apply(regress, 'Power', ['Gas','Coal','Carbon'])



# 2. Inserting date in the DataFrame
d = pd.date_range('2010/01/01','2012/12/31',freq='B')
dd = d.day
mm = d.month
yy = d.year

xls_data.insert(0,'Day',dd)
xls_data.insert(1,'Month',mm)
xls_data.insert(2,'Year',yy)

prices_y = xls_data.set_index(['Year'])
prices_m = xls_data.set_index(['Month'])

# 3. Transformation in relative returns
lprices = xls_data.pct_change()
lprices_y = lprices.set_index(['Year'])
lprices_m = lprices.set_index(['Month'])


# 4. Cleaning data (it has to be the last...)
prices_y_na = prices_y.dropna()
# prices_na = xls_data.dropna()
prices_m_na = prices_m.dropna()
lprices_m_na = lprices_m.dropna()

# returns = prices.pct_change()
# returns = returns.dropna()

# 5. Grouping data

# lprices_na_by_month = lprices_m_na.groupby('Month') will not work 
# lprices_by_month = xls_data.groupby('Month')
# lprices_na_by_month =lprices_by_month.dropna()

# 6. Creating arrays for the different regressions

# 6.1. Yearly arrays
first_y  = prices_y_na.ix['2010']
second_y = prices_y_na.ix['2011']
third_y  = prices_y_na.ix['2012']

# 6.2. Montly arrays
# Jan_p = prices_m_na.ix['1']
# Feb_p = prices_m_na.ix['2']
# Mar_p = prices_m_na.ix['3']
# Apr_p = prices_m_na.ix['4']
# May_p = prices_m_na.ix['5']
# Jun_p = prices_m_na.ix['6']
# Jul_p = prices_m_na.ix['7'] 
# Ago_p = prices_m_na.ix['8']
# Sep_p = prices_m_na.ix['9']
# Oct_p = prices_m_na.ix['10']
# Nov_p = prices_m_na.ix['11']
# Dec_p = prices_m_na.ix['12']


