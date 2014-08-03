# ======================================================================
# Quantitative Strategy (QS1/2) Stats. Reasearch Part - Python Script
# 
# Author: 		Jesus Perez Colino 
# Date: 		Starting day: 10 of April of 2013
# Description: 	This scripts contains all the statistics needed for 
# 				the analysis and research of the QS1 and QS2
# ======================================================================
# ======================================================================
from pandas import Series, DataFrame,isnull, ExcelFile, date_range, rolling_corr, ols, rolling_corr_pairwise, rolling_std
from pylab import plot, show
import pandas as pd
import pylab as pl
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
# ======================================================================
# 0. Initialization of parameters for all the plots
# ======================================================================
plt.close('all')
plt.rc('axes',grid=True)
plt.rc('grid',color='0.75',linestyle='-',linewidth=0.75)
# ======================================================================
# 1. Importing data
# ======================================================================
path_xls = 'C:\Users\Suso\Desktop\Ipython Notebooks\Naivereg2.csv'
xls_data = pd.read_csv(path_xls, parse_dates=True, index_col=0)
# ======================================================================
# 2. Data transformation and cleaning (Winsorization)
# ======================================================================
prices = xls_data.dropna() # we have now 1240 'clean' prices from 2008
start = prices.index[0]
end = prices.index[prices.index.size-1]
period_d = date_range(start, periods = xls_data.index.size, freq = 'B')
log_prices = DataFrame(prices.pct_change(), index=prices.index)
# ----------------------------------------------------------------------
# Cleaning outliers from data by winsorization
# ---------------------------------------------------------------------- 
w_prices = log_prices.copy()
std_1y = rolling_std(log_prices, 250, min_periods=20)
cap_level = 3 * np.sign(w_prices) * std_1y
w_prices[np.abs(w_prices) > 3*std_1y] = cap_level
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# FIGURE 1
prices.plot(subplots=True, figsize=(8,8), style='b-',grid=True)
# plt.title('Energy Commodities Prices') 
plt.legend(loc='best')
plt.grid(True)
# ----------------------------------------------------------------------
# FIGURE 2 
# ----------------------------------------------------------------------
log_prices.plot(subplots=True, figsize=(8,8), style='b-',grid=True)
plt.legend(loc='best')
# ----------------------------------------------------------------------
# FIGURE 3
# ----------------------------------------------------------------------
pd.scatter_matrix(log_prices, alpha=0.2, figsize=(8,8), diagonal='kde')
# ----------------------------------------------------------------------
# FIGURE 4
# ----------------------------------------------------------------------
w_prices.plot(subplots=True, figsize=(8,8),style='b-', grid=True)
plt.legend(loc='best')
# ----------------------------------------------------------------------
# FIGURE 5
# ----------------------------------------------------------------------
pd.scatter_matrix(w_prices, alpha=0.2, figsize=(8,8), diagonal='kde')
# ======================================================================
# 3. Basic Statisitics and Plots functions
# ======================================================================
# ----------------------------------------------------------------------
# FIGURE 6 : Rolling Correlation between Electricity and ...
# ----------------------------------------------------------------------
fig6 = plt.figure()

ax01 = fig6.add_subplot(3,2,1)
ax02 = fig6.add_subplot(3,2,3)
ax03 = fig6.add_subplot(3,2,5)
ax04 = fig6.add_subplot(3,2,2)
ax05 = fig6.add_subplot(3,2,4)
ax06 = fig6.add_subplot(3,2,6)

rcorr1eg25 = Series(rolling_corr(prices.Electricity.values, prices.Gas.values, 25), index = prices.index)
rcorr2eg75 = Series(rolling_corr(prices.Electricity.values, prices.Gas.values, 75), index = prices.index)
w_rcorr1eg25 = Series(rolling_corr(w_prices.Electricity.values, w_prices.Gas.values, 25), index = w_prices.index)
w_rcorr2eg75 = Series(rolling_corr(w_prices.Electricity.values, w_prices.Gas.values, 75), index = w_prices.index)

rcorr1ec25 = Series(rolling_corr(prices.Electricity.values, prices.Coal.values, 25), index = prices.index)
rcorr2ec75 = Series(rolling_corr(prices.Electricity.values, prices.Coal.values, 75), index = prices.index)
w_rcorr1ec25 = Series(rolling_corr(w_prices.Electricity.values, w_prices.Coal.values, 25), index = w_prices.index)
w_rcorr2ec75 = Series(rolling_corr(w_prices.Electricity.values, w_prices.Coal.values, 75), index = w_prices.index)

rcorr1ecb25 = Series(rolling_corr(prices.Electricity.values, prices.Carbon.values, 25), index = prices.index)
rcorr2ecb75 = Series(rolling_corr(prices.Electricity.values, prices.Carbon.values, 75), index = prices.index)
w_rcorr1ecb25 = Series(rolling_corr(w_prices.Electricity.values, w_prices.Carbon.values, 25), index = w_prices.index)
w_rcorr2ecb75 = Series(rolling_corr(w_prices.Electricity.values, w_prices.Carbon.values, 75), index = w_prices.index)

ax01.plot(rcorr1eg25.index, rcorr1eg25.values, 'k--', label = 'Corr25')
ax01.plot(rcorr2eg75.index, rcorr2eg75.values, 'ro-', label = 'Corr75')
ax01.legend(loc='best')
ax01.set_xlim([start,end])
ax01.set_title('Rolling Correlation between Electricity and Gas')

ax02.plot(rcorr1ec25.index, rcorr1ec25.values, 'k--', label = 'Corr25')
ax02.plot(rcorr2ec75.index, rcorr2ec75.values, 'bo-', label = 'Corr75')
ax02.legend(loc='best')
ax02.set_xlim([start,end])
ax02.set_title('Rolling Correlation between Electricity and Coal')

ax03.plot(rcorr1ecb25.index, rcorr1ecb25.values, 'k--', label = 'Corr25')
ax03.plot(rcorr2ecb75.index, rcorr2ecb75.values, 'go-', label = 'Corr75')
ax03.legend(loc='best')
ax03.set_xlim([start,end])
ax03.set_title('Roll.Correlation between Electricity and Carbon')

ax04.plot(w_rcorr1eg25.index, w_rcorr1eg25.values, 'k--', label = 'Log_WCorr25')
ax04.plot(w_rcorr2eg75.index, w_rcorr2eg75.values, 'ro-', label = 'Log_WCorr75')
ax04.legend(loc='best')
ax04.set_xlim([start,end])
ax04.set_title('Roll.Correlation between Electricity and Gas')

ax05.plot(w_rcorr1ec25.index, w_rcorr1ec25.values, 'k--', label = 'Log_WCorr25')
ax05.plot(w_rcorr2ec75.index, w_rcorr2ec75.values, 'bo-', label = 'Log_WCorr75')
ax05.legend(loc='best')
ax05.set_xlim([start,end])
ax05.set_title('Roll.Correlation between Electricity and Coal')

ax06.plot(w_rcorr1ecb25.index, w_rcorr1ecb25.values, 'k--', label = 'Log_WCorr25')
ax06.plot(w_rcorr2ecb75.index, w_rcorr2ecb75.values, 'go-', label = 'Log_WCorr75')
ax06.legend(loc='best')
ax06.set_xlim([start,end])
ax06.set_title('Rolling Correlation between Electricity and Carbon')

del(rcorr1eg25, rcorr2eg75, w_rcorr1eg25, w_rcorr2eg75)
del(rcorr1ec25, rcorr2ec75, w_rcorr1ec25, w_rcorr2ec75)
del(rcorr1ecb25, rcorr2ecb75, w_rcorr1ecb25, w_rcorr2ecb75)
# ----------------------------------------------------------------------
# FIGURE 7 : Rolling beta and intercept of the regression...
# ----------------------------------------------------------------------
fig7 = plt.figure()

ax01 = fig7.add_subplot(3,2,1)
ax02 = fig7.add_subplot(3,2,3)
ax03 = fig7.add_subplot(3,2,5)
ax04 = fig7.add_subplot(3,2,2)
ax05 = fig7.add_subplot(3,2,4)
ax06 = fig7.add_subplot(3,2,6)

beta1eg25 = ols(y=prices.Electricity, x=prices.Gas, window=25).beta
beta2eg75 = ols(y=prices.Electricity, x=prices.Gas, window=75).beta
beta1ec25 = ols(y=prices.Electricity, x=prices.Coal, window=25).beta
beta2ec75 = ols(y=prices.Electricity, x=prices.Coal, window=75).beta
beta1ecb25 = ols(y=prices.Electricity, x=prices.Carbon, window=25).beta
beta2ecb75 = ols(y=prices.Electricity, x=prices.Carbon, window=75).beta

w_beta1eg25 = ols(y=w_prices.Electricity, x=w_prices.Gas, window=25).beta
w_beta2eg75 = ols(y=w_prices.Electricity, x=w_prices.Gas, window=75).beta
w_beta1ec25 = ols(y=w_prices.Electricity, x=w_prices.Coal, window=25).beta
w_beta2ec75 = ols(y=w_prices.Electricity, x=w_prices.Coal, window=75).beta
w_beta1ecb25 = ols(y=w_prices.Electricity, x=w_prices.Carbon, window=25).beta
w_beta2ecb75 = ols(y=w_prices.Electricity, x=w_prices.Carbon, window=75).beta

ax01.plot(beta1eg25.x.index, beta1eg25.x.values, 'k--', label = 'Beta25')
ax01.plot(beta2eg75.x.index, beta2eg75.x.values, 'ro-', label = 'Beta75')
ax01.legend(loc='best')
ax01.set_xlim([start,end])
ax01.set_title('Rolling Beta between Electricity and Gas')

ax02.plot(beta1ec25.x.index, beta1ec25.x.values, 'k--', label = 'Beta25')
ax02.plot(beta2ec75.x.index, beta2ec75.x.values, 'bo-', label = 'Beta75')
ax02.legend(loc='best')
ax02.set_xlim([start,end])
ax02.set_title('Rolling Beta between Electricity and Coal')

ax03.plot(beta1ecb25.x.index, beta1ecb25.x.values, 'k--', label = 'Beta25')
ax03.plot(beta2ecb75.x.index, beta2ecb75.x.values, 'go-', label = 'Beta75')
ax03.legend(loc='best')
ax03.set_xlim([start,end])
ax03.set_title('Rolling Beta between Electricity and Carbon')

ax04.plot(w_beta1eg25.intercept.index, w_beta1eg25.intercept.values, 'k--', label = 'Beta25')
ax04.plot(w_beta2eg75.intercept.index, w_beta2eg75.intercept.values, 'ro-', label = 'Beta75')
ax04.legend(loc='best')
ax04.set_xlim([start,end])
ax04.set_title('Roll.Beta between (log_w) Electricity and Gas')

ax05.plot(w_beta1ec25.intercept.index, w_beta1ec25.intercept.values, 'k--', label = 'Beta25')
ax05.plot(w_beta2ec75.intercept.index, w_beta2ec75.intercept.values, 'bo-', label = 'Beta75')
ax05.legend(loc='best')
ax05.set_xlim([start,end])
ax05.set_title('Roll.Intercept between Electricity and Coal')

ax06.plot(w_beta1ecb25.intercept.index, w_beta1ecb25.intercept.values, 'k--', label = 'Beta25')
ax06.plot(w_beta2ecb75.intercept.index, w_beta2ecb75.intercept.values, 'go-', label = 'Beta75')
ax06.legend(loc='best')
ax06.set_xlim([start,end])
ax06.set_title('Roll.Intercept between Electricity and Carbon')

del(beta1eg25, beta2eg75, beta1ec25, beta2ec75, beta1ecb25, beta2ecb75)
del(w_beta1eg25, w_beta2eg75, w_beta1ec25, w_beta2ec75, w_beta1ecb25, w_beta2ecb75)
# ----------------------------------------------------------------------
# FIGURE 8 : Rolling beta and intercept for the full regression
# ----------------------------------------------------------------------
fig8 = plt.figure()

ax01 = fig8.add_subplot(4,2,1)
ax02 = fig8.add_subplot(4,2,3)
ax03 = fig8.add_subplot(4,2,5)
ax04 = fig8.add_subplot(4,2,7)
ax05 = fig8.add_subplot(4,2,2)
ax06 = fig8.add_subplot(4,2,4)
ax07 = fig8.add_subplot(4,2,6)
ax08 = fig8.add_subplot(4,2,8)

reg1e25 = ols(y=prices.Electricity, x={'Gas':prices.Gas, 'Coal':prices.Coal, 'Carbon':prices.Carbon}, window=25).beta
reg2e75 = ols(y=prices.Electricity, x={'Gas':prices.Gas, 'Coal':prices.Coal, 'Carbon':prices.Carbon}, window=75).beta

w_reg1e25 = ols(y=w_prices.Electricity, x={'Gas':w_prices.Gas, 'Coal':w_prices.Coal, 'Carbon':w_prices.Carbon}, window=25).beta
w_reg2e75 = ols(y=w_prices.Electricity, x={'Gas':w_prices.Gas, 'Coal':w_prices.Coal, 'Carbon':w_prices.Carbon}, window=75).beta

ax01.plot(reg1e25.intercept.index, reg1e25.intercept.values,'k--', label = 'intercept25')
ax01.plot(reg2e75.intercept.index, reg2e75.intercept.values,'ro-', label = 'intercept75')
ax01.set_title('Rolling Intercept in full regression')
ax01.legend(loc='best')

ax02.plot(reg1e25.Gas.index, reg1e25.Gas.values,'k--', label = 'gas25')
ax02.plot(reg2e75.Gas.index, reg2e75.Gas.values,'co-', label = 'gas75')
ax02.set_title('Rolling Beta for Gas in full regression')
ax02.legend(loc='best')

ax03.plot(reg1e25.Coal.index, reg1e25.Coal.values,'k--', label = 'coal25')
ax03.plot(reg2e75.Coal.index, reg2e75.Coal.values,'bo-', label = 'coal75')
ax03.set_title('Rolling Beta for Coal in full regression')
ax03.legend(loc='best')

ax04.plot(reg1e25.Carbon.index, reg1e25.Carbon.values,'k--', label = 'carbon25')
ax04.plot(reg2e75.Carbon.index, reg2e75.Carbon.values,'go-', label = 'carbon75')
ax04.set_title('Rolling Beta for Carbon in full regression')
ax04.legend(loc='best')

ax05.plot(w_reg1e25.intercept.index, w_reg1e25.intercept.values,'k--', label = 'intercept25')
ax05.plot(w_reg2e75.intercept.index, w_reg2e75.intercept.values,'ro-', label = 'intercept75')
ax05.set_title('Rolling Intercept in full regression')
ax05.legend(loc='best')

ax06.plot(w_reg1e25.Gas.index, w_reg1e25.Gas.values,'k--', label = 'gas25')
ax06.plot(w_reg2e75.Gas.index, w_reg2e75.Gas.values,'co-', label = 'gas75')
ax06.set_title('Rolling Beta for Gas in full regression')
ax06.legend(loc='best')

ax07.plot(w_reg1e25.Coal.index, w_reg1e25.Coal.values,'k--', label = 'coal25')
ax07.plot(w_reg2e75.Coal.index, w_reg2e75.Coal.values,'bo-', label = 'coal75')
ax07.set_title('Rolling Beta for Coal in full regression')
ax07.legend(loc='best')

ax08.plot(w_reg1e25.Carbon.index, w_reg1e25.Carbon.values,'k--', label = 'carbon25')
ax08.plot(w_reg2e75.Carbon.index, w_reg2e75.Carbon.values,'go-', label = 'carbon75')
ax08.set_title('Rolling Beta for Carbon in full regression')
ax08.legend(loc='best')

del (reg1e25, reg2e75, w_reg1e25, w_reg2e75)
# ----------------------------------------------------------------------
# FIGURE 9 : Rolling Correlations in ful regressions ... 
# ----------------------------------------------------------------------
fig9 = plt.figure()
ax01 = fig9.add_subplot(3,2,1)
ax02 = fig9.add_subplot(3,2,3)
ax03 = fig9.add_subplot(3,2,5)
ax04 = fig9.add_subplot(3,2,2)
ax05 = fig9.add_subplot(3,2,4)
ax06 = fig9.add_subplot(3,2,6)

correls25 = rolling_corr_pairwise(prices, 25)
correls75 = rolling_corr_pairwise(prices, 75)
correls250 = rolling_corr_pairwise(prices, 250)

w_correls25 = rolling_corr_pairwise(w_prices, 25)
w_correls75 = rolling_corr_pairwise(w_prices, 75)
w_correls250 = rolling_corr_pairwise(w_prices, 250)

ax01.plot( correls25.ix[:,'Electricity', 'Gas'], 'k--', label = 'corr25' )
ax01.plot( correls75.ix[:,'Electricity', 'Gas'], 'ro-', label = 'corr75' )
ax01.plot( correls250.ix[:,'Electricity', 'Gas'], 'r*-', label = 'corr250' )
ax01.set_title('Rolling Correlation for Gas in full regression')
ax01.legend(loc='best')
ax01.set_xlim([start,end])

ax02.plot( correls25.ix[:,'Electricity', 'Coal'], 'k--', label = 'corr25' )
ax02.plot( correls75.ix[:,'Electricity', 'Coal'], 'bo-', label = 'corr75' )
ax02.plot( correls250.ix[:,'Electricity', 'Coal'], 'b*-', label = 'corr250' )
ax02.set_title('Rolling Correlation for Coal in full regres.')
ax02.legend(loc='best')

ax03.plot( correls25.ix[:,'Electricity', 'Carbon'], 'k--', label = 'corr25' )
ax03.plot( correls75.ix[:,'Electricity', 'Carbon'], 'go-', label = 'corr75' )
ax03.plot( correls250.ix[:,'Electricity', 'Carbon'], 'g*-', label = 'corr250' )
ax03.set_title('Rolling Correlation for Carbon in full regres.')
ax03.legend(loc='best')

ax04.plot( w_correls25.ix[:,'Electricity', 'Gas'], 'k--', label = 'corr25' )
ax04.plot( w_correls75.ix[:,'Electricity', 'Gas'], 'ro-', label = 'corr75' )
ax04.plot( w_correls250.ix[:,'Electricity', 'Gas'], 'r*-', label = 'corr250' )
ax04.set_title('Roll.Correlation for Gas in full regres.')
ax04.legend(loc='best')

ax05.plot( w_correls25.ix[:,'Electricity', 'Coal'], 'k--', label = 'corr25' )
ax05.plot( w_correls75.ix[:,'Electricity', 'Coal'], 'bo-', label = 'corr75' )
ax05.plot( w_correls250.ix[:,'Electricity', 'Coal'], 'b*-', label = 'corr250' )
ax05.set_title('Roll.Correlation for Coal in full regres.')
ax05.legend(loc='best')

ax06.plot( w_correls25.ix[:,'Electricity', 'Carbon'], 'k--', label = 'corr25' )
ax06.plot( w_correls75.ix[:,'Electricity', 'Carbon'], 'go-', label = 'corr75' )
ax06.plot( w_correls250.ix[:,'Electricity', 'Carbon'], 'g*-', label = 'corr250' )
ax06.set_title('Roll.Correlation for Carbon in full regres.')
ax06.legend(loc='best')

# ----------------------------------------------------------------------
# FIGURE 10
# ----------------------------------------------------------------------
fig10 = plt.figure()
ax01 = fig10.add_subplot(3,1,1)
ax02 = fig10.add_subplot(3,1,2)
ax03 = fig10.add_subplot(3,1,3)

ax01.plot(log_prices.index, rolling_std(w_prices.values,75))
ax01.legend(loc='best')
ax02.plot( correls75.ix[:,'Electricity', 'Gas'], 'ro-', label = 'corr75_wGas' )
ax02.plot( correls75.ix[:,'Electricity', 'Coal'], 'bo-', label = 'corr75_wCoal' )
ax02.legend(loc='best')

del(correls25,correls75, correls250)
del(w_correls25,w_correls75, w_correls250)

# plt.title('Rolling Correlation for Gas and Coal in full regression')
# fig6.plot( correls250.ix[:,'Electricity', 'Gas'], 'ro-', label = 'corr250' )
# fig6.plot( correls250.ix[:,'Electricity', 'Coal'], 'bo-', label = 'corr250' )
# fig6.plot( correls250.ix[:,'Electricity', 'Carbon'], 'go-', label = 'corr250' )
# ======================================================================
# 4. Yearly regressions
# ======================================================================

commdty_corr = lambda x: x.corrwith(x['Electricity'])
commdty_std = lambda x: x.std()
by_year = prices.groupby(lambda x: x.year)
by_month = prices.groupby(lambda x: x.month)
def regress(data,yvar,xvars):
    Y = data[yvar]
    X = data[xvars]
    X['intercept'] = 1.
    result = sm.OLS(Y,X).fit()
	# residual1 = result.resid
    print ' '
    print '*'*80
    print result.summary()
    return result.params 
	#, residual1
    
def residual(data,yvar,xvars):
    Y = data[yvar]
    X = data[xvars]
    X['intercept'] = 1.
    residual = sm.OLS(Y,X).fit().resid
    # print result.summary
    return residual

print 'Basic Stats. description by year'
print '='*80
print by_year.describe()
print '='*80
print 'Std Table per year'
print '='*80
print by_year.apply(commdty_std)
print '='*80
print 'Std Table per month'
print '='*80
print by_month.apply(commdty_std)
print '='*80
print 'Correlation Table per year'
print '='*80
print by_year.apply(commdty_corr)
print '='*80
print 'Correlation Table per month'
print '='*80
print by_month.apply(commdty_corr)
print '='*80

results1 = by_year.apply(regress, 'Electricity', ['Gas','Coal','Carbon'])
results2 = by_year.apply(residual, 'Electricity', ['Gas','Coal','Carbon'])

print '='*80
print 'Coefficientes of the Regression ' 
print '-'*80
print results1 #['parameters']
print '-'*80

# ======================================================================
# 5. Ploting of the histograms of residuals 
# ======================================================================
# ----------------------------------------------------------------------
# FIGURE 11
# ----------------------------------------------------------------------
fig11 = plt.figure()
ax01 = fig11.add_subplot(3,2,1)
ax02 = fig11.add_subplot(3,2,2)
ax03 = fig11.add_subplot(3,2,3)
ax04 = fig11.add_subplot(3,2,4)
ax05 = fig11.add_subplot(3,2,5)
ax06 = fig11.add_subplot(3,2,6)

ax01.hist(results2[2008],bins=25, alpha = 0.2, color='g', normed=True)
ax01.set_title('2008')
ax02.hist(results2[2009],bins=25, alpha = 0.2, color='c', normed=True)
ax02.set_title('2009')
ax03.hist(results2[2010],bins=25, alpha = 0.2, color='b', normed=True)
ax03.set_title('2010')
ax04.hist(results2[2011],bins=25, alpha = 0.2, color='k', normed=True)
ax04.set_title('2011')
ax05.hist(results2[2012],bins=25, alpha = 0.2, color='r', normed=True)
ax05.set_title('2012')
ax06.hist(results2[2013],bins=25, alpha = 0.2, color='y', normed=True)
ax06.set_title('2013')

fig11.text(0.5,0.975,'OLS-Fuel Regress: Histogram of residuals in-the-sample'
, horizontalalignment='center', verticalalignment='top')
# ======================================================================
# 6. Calculation and Ploting of the prediction out-of-sample
# ======================================================================
data2 = prices.set_index(prices.index.year)
prediction = DataFrame(np.zeros(data2.Electricity.count()),index=prices.index)
seq_years = range(results1.index[0],results1.index[-1])		
predict_out = np.zeros(data2.Electricity.count())
results3 = np.zeros(1)
for i in seq_years : 
	year_p = np.array(data2.ix[i+1])
	year_seq = range(0 , data2.ix[i+1].Electricity.count())
	# predict_out 
	for j in year_seq :
		predict_out[j] = results1.ix[i]['intercept'] + results1.ix[i]['Gas'] * year_p[j,1] \
		+ results1.ix[i]['Coal']*year_p[j,2] + results1.ix[i]['Carbon']*year_p[j,3] 
		results3 = np.append(results3,predict_out[j])

nn = np.zeros(data2.Electricity.count()-results3.shape)
nn = np.append(nn,results3)
prediction = DataFrame(nn[data2.Electricity.count()-results3.shape:], 
index = prices.index[data2.Electricity.count()-results3.shape:])
real_data = DataFrame(data2.Electricity.values[data2.Electricity.count()-results3.shape:], 
index = prices.index[data2.Electricity.count()-results3.shape:])
residual_out = prediction[1:] - real_data[1:]

# ----------------------------------------------------------------------
# FIGURE 12 : Residuals Out-of-the-Sample
# ----------------------------------------------------------------------
fig12 = plt.figure()
ax01 = fig12.add_subplot(3,1,1)
ax02 = fig12.add_subplot(3,1,2)
ax03 = fig12.add_subplot(3,1,3)

ax01.plot( prediction.index[1:], prediction.values[1:], label = 'Prediction')
ax01.plot(real_data.index[1:], real_data.values[1:], label = 'Current prices')
ax01.legend()
ax01.set_title('Actual prices vs. Out-The-Sample Prediction')

ax02.plot(residual_out.index, residual_out.values ,'ro--', label = 'Residual out-the-sample')
ax02.legend()
ax02.set_title('Residuals per year')

ax03.hist(residual_out.values, bins=25, alpha = 0.2, color='b', normed=True)
ax03.set_title('Histogram of Residuals')
fig12.text(0.5,0.975,'Actual vs Predictions out-the-sample', horizontalalignment='center',
verticalalignment='top')
fillcolor = 'darkslategrey'

# ======================================================================
# 7. BACKTESTING OF THE REGRESSION-RESIDUALS STRATEGY
# ======================================================================

# ----------------------------------------------------------------------
# 7.1. Splititing in vectors of years the results of by_year_resid
# ----------------------------------------------------------------------

by_year_resid = residual_out.groupby(lambda x: x.year)

power_y = DataFrame(prices.Electricity.values, index=prices.index.year)
gas_y = DataFrame(prices.Gas.values, index=prices.index.year)
coal_y = DataFrame(prices.Coal.values, index=prices.index.year)
carbon_y = DataFrame(prices.Carbon.values, index=prices.index.year)

res_y = DataFrame(residual_out.values[:,0], index=residual_out.index.year)

years = np.arange(start.year + 1, end.year, 1)
size_n = np.zeros(1)

pr_year = prices.set_index(prices.index.year)
pr_year = pr_year.Electricity
pr_year = pr_year.groupby(level = 0)


for j in years :
	num = np.size(by_year_resid.groups[j])
	size_n = np.append(size_n, num)
	
size_n = np.cumsum(size_n)
res_09 = np.array(res_y[size_n[0]:size_n[1]].values)
res_10 = np.array(res_y[size_n[1]+1:size_n[2]].values)
res_11 = np.array(res_y[size_n[2]+1:size_n[3]].values)
res_12 = np.array(res_y[size_n[3]+1:size_n[4]].values)
tt = np.size(res_y)
res_13 = np.array(res_y[size_n[4]+1:tt].values)

# ----------------------------------------------------------------------
# 7.2. Running the Moving Average in the residuals of each year
# ----------------------------------------------------------------------

def moving_average(x, n, type='simple'):
    x = np.asarray(x)
    if type=='simple':
        weights = np.ones(n)
    else:
        weights = np.exp(np.linspace(-1., 0., n))
    weights /= weights.sum()
    a =  np.convolve(x, weights, mode='full')[:len(x)]
    a[:n] = a[n]
    return a

ma_res_09 = moving_average(res_09[:,0], 20,  type='simple')
std_res09 = rolling_std(res_09[:,0], 20)
ma_res_10 = moving_average(res_10[:,0], 20,  type='simple') 
std_res10 = rolling_std(res_10[:,0], 20)
ma_res_11 = moving_average(res_11[:,0], 20,  type='simple')  
std_res11 = rolling_std(res_11[:,0], 20)
ma_res_12 = moving_average(res_12[:,0], 20,  type='simple')
std_res12 = rolling_std(res_12[:,0], 20)    	
ma_res_13 = moving_average(res_13[:,0], 20,  type='simple')
std_res13 = rolling_std(res_13[:,0], 20)  

# ----------------------------------------------------------------------
# FIGURE 13 : Residuals Out-of-the-Sample (MA and Std)
# ----------------------------------------------------------------------
fig13 = plt.figure()
ax01 = fig13.add_subplot(4,1,1)
ax02 = fig13.add_subplot(4,1,2)
ax03 = fig13.add_subplot(4,1,3)
ax04 = fig13.add_subplot(4,1,4)

num_std = 1

ax01.plot( ma_res_09,'k--', label='MA25_resid' )
ax01.plot( ma_res_09 + num_std*std_res09, 'r--', label='MA25+STD25_resid' )
ax01.plot( ma_res_09 - num_std*std_res09, 'r--', label='MA25+STD25_resid' )
ax01.plot( res_09[:,0], 'ro--', label='MA25+STD25_resid' )
ax05 = ax01.twinx()
ax05.plot(pr_year.values[2009], 'k-')
#ax01.legend(loc='best')
ax01.set_title('Residuals2009')

ax02.plot( ma_res_10, 'k--', label='MA25_resid' )
ax02.plot( ma_res_10 + num_std*std_res10, 'r--', label='MA25+STD25_resid' )
ax02.plot( ma_res_10 - num_std*std_res10, 'r--', label='MA25+STD25_resid' )
ax02.plot( res_10[:,0], 'ro--', label='MA25+STD25_resid' )
ax05 = ax02.twinx()
ax05.plot(pr_year.values[2010], 'k-')
#ax02.legend(loc='best')
ax02.set_title('Residuals2010')

ax03.plot( ma_res_11, 'k--', label='MA25_resid' )
ax03.plot( ma_res_11 + num_std*std_res11, 'r--', label='MA25+STD25_resid' )
ax03.plot( ma_res_11 - num_std*std_res11, 'r--', label='MA25+STD25_resid' )
ax03.plot( res_11[:,0], 'ro--', label='MA25+STD25_resid' )
ax05 = ax03.twinx()
ax05.plot(pr_year.values[2011], 'k-')
#ax03.legend(loc='best')
ax03.set_title('Residuals2011')

ax04.plot( ma_res_12,'k--', label='MA25_resid' )
ax04.plot( ma_res_12 + num_std*std_res12, 'r--', label='MA25+STD25_resid' )
ax04.plot( ma_res_12 - num_std*std_res12, 'r--', label='MA25+STD25_resid' )
ax04.plot( res_12[:,0], 'ro--', label='MA25+STD25_resid' )
ax05 = ax04.twinx()
ax05.plot(pr_year.values[2012], 'k-')
#ax04.legend(loc='best')
ax04.set_title('Residuals2012')

# ----------------------------------------------------------------------
# 7.3. Running the Trading Rule, store the results and plot-it
# ----------------------------------------------------------------------

def position(year, p_power, p_gas, p_coal, p_carbon, resid, num_std, ma_resid, std_resid):

	p_power = np.asarray(p_power)
	p_gas = np.asarray(p_gas)
	p_coal = np.asarray(p_coal)
	p_carbon = np.asarray(p_carbon)
	
	n = len(p_power)
	position_f = np.zeros(n) 	# Final realized result 
	position_m2m = np.zeros(n) 	# Mark-to-market trading position
	results = np.zeros(n)
	
	# REMEMBER: resid = prediction - real_data
	#			if resid > 0 : Market is BEARISH and MARKET IS RIGHT (we follow the market feeling) 
	# 						  -> We are BEARISH and we go SHORT
	#			if resid < 0 : Market is BULLISH and MARKET IS RIGHT (we follow the market feeling)
	# 						  -> We are BULLISH and we go LONG
	for k in range(1,n-1):
	
		# OPENING POSITIONS -----------------------------------------------------

		if position_f[k-1] == 0 and resid[k] < ma_resid[k] - num_std * std_resid[k] \
		and resid[k] < 0 :
			position_f[k] =  p_power[k] - results1.Gas.ix[year] * p_gas[k]\
			    - results1.Carbon.ix[year] * p_carbon[k]	# Long position if resid < 0 
			position_m2m[k] = position_m2m[k-1]	
		
		elif position_f[k-1] == 0 and resid[k] > ma_resid[k] + num_std * std_resid[k] \
		and resid[k] < 0 :
			position_f[k] = - p_power[k] + results1.Gas.ix[year] * p_gas[k]\
			    + results1.Carbon.ix[year] * p_carbon[k] # Short position if resid < 0 
			position_m2m[k] = position_m2m[k-1]
		
		elif position_f[k-1] == 0 and resid[k] > ma_resid[k] + num_std * std_resid[k] \
		and resid[k] > 0 :
			position_f[k] =  - p_power[k] + results1.Gas.ix[year] * p_gas[k]\
			    + results1.Carbon.ix[year] * p_carbon[k] # Short position if resid > 0 
			position_m2m[k] = position_m2m[k-1]		
		
		elif position_f[k-1] == 0 and resid[k] < ma_resid[k] - num_std * std_resid[k] \
		and resid[k] > 0 :
			position_f[k] =  p_power[k] - results1.Gas.ix[year] * p_gas[k]\
			    - results1.Carbon.ix[year] * p_carbon[k] # Long position if resid > 0 
			position_m2m[k] = position_m2m[k-1]	
		
		elif position_f[k-1] == 0 :
			position_m2m[k] = position_m2m[k-1]
			
		# WAITING ----------------------------------------------------------------------
		
		# WAITING in a SHORT position (from a BEARISH mkt <-> resid > 0)
		if position_f[k-1] < 0 and resid[k] > 0 and resid[k] > ma_resid[k] - num_std * std_resid[k]  :
			position_f[k] = position_f[k-1]				# Doing nothing w. Short position
			position_m2m[k] = position_m2m[k-1] + p_power[k-1] - results1.Gas.ix[year] * p_gas[k-1] \
			    - results1.Carbon.ix[year] * p_carbon[k-1] \
			    - p_power[k] + results1.Gas.ix[year] * p_gas[k] + results1.Carbon.ix[year] * p_carbon[k]	
		
		# WAITING in a SHORT position (from a BULLISH mkt <-> resid < 0)
		elif position_f[k-1] < 0 and resid[k] < 0 and resid[k] > ma_resid[k] - num_std * std_resid[k]  :
			position_f[k] = position_f[k-1]				# Doing nothing w. Short position
			position_m2m[k] = position_m2m[k-1] + p_power[k-1] - results1.Gas.ix[year] * p_gas[k-1] \
			    - results1.Carbon.ix[year] * p_carbon[k-1] \
			    - p_power[k] + results1.Gas.ix[year] * p_gas[k] + results1.Carbon.ix[year] * p_carbon[k]			
		
		# WAITING in a LONG position (from a BULLISH mkt <-> resid < 0)
		elif position_f[k-1] > 0 and resid[k] > 0 and resid[k] < ma_resid[k] + num_std * std_resid[k]  :
			position_f[k] = position_f[k-1]				# Doing nothing w. Long position
			position_m2m[k] = position_m2m[k-1] + p_power[k] - results1.Gas.ix[year] * p_gas[k] \
			    - results1.Carbon.ix[year] * p_carbon[k] \
			    - p_power[k-1] + results1.Gas.ix[year] * p_gas[k-1] + results1.Carbon.ix[year] * p_carbon[k-1]			
			
		# WAITING in a LONG position (from a BEARISH mkt <-> resid > 0)	
		elif position_f[k-1] > 0 and resid[k] < 0 and resid[k] < ma_resid[k] + num_std * std_resid[k]  :
			position_f[k] = position_f[k-1]				# Doing nothing w. Long position
			position_m2m[k] = position_m2m[k-1] + p_power[k] - results1.Gas.ix[year] * p_gas[k] \
			    - results1.Carbon.ix[year] * p_carbon[k] \
			    - p_power[k-1] + results1.Gas.ix[year] * p_gas[k-1] + results1.Carbon.ix[year] * p_carbon[k-1]					
			
		# CLOSING POSITIONS ----------------------------------------------------------------------------
		
		# Closing a SHORT position (from a BEARISH mkt <-> resid > 0)
		if position_f[k-1] < 0 and resid[k] < ma_resid[k] - num_std * std_resid[k]  :
			results[k] = - position_f[k-1] - p_power[k] + results1.Gas.ix[year] * p_gas[k] \
			    + results1.Carbon.ix[year] * p_carbon[k]  
			position_f[k] = 0
			position_m2m[k] = position_m2m[k-1] + p_power[k-1] - results1.Gas.ix[year] * p_gas[k-1] \
			    - results1.Carbon.ix[year] * p_carbon[k-1] \
			    - p_power[k] + results1.Gas.ix[year] * p_gas[k] + results1.Carbon.ix[year] * p_carbon[k]
			
		# CLOSING a SHORT position (from a BULLISH mkt <-> resid < 0)
		# elif position_f[k-1] < 0 and resid [k]<0 and resid[k] < ma_resid[k] - num_std * std_resid[k]  :
			# results[k] = - position_f[k-1] - prces[k] 	
			# position_f[k] = 0
			# position_m2m[k] = position_m2m[k-1] + prces[k-1] - prces[k]
		
		# Closing LONG position	(from a BULLISH mkt <-> resid < 0)
		elif position_f[k-1] > 0 and resid[k] > ma_resid[k] + num_std * std_resid[k]  :	
			results[k] = p_power[k] - results1.Gas.ix[year] * p_gas[k] \
			    - results1.Carbon.ix[year] * p_carbon[k] - position_f[k-1] 	
			position_f[k] = 0
			position_m2m[k] = position_m2m[k-1] + p_power[k] - results1.Gas.ix[year] * p_gas[k] \
			    - results1.Carbon.ix[year] * p_carbon[k] \
			    - p_power[k-1] + results1.Gas.ix[year] * p_gas[k-1] + results1.Carbon.ix[year] * p_carbon[k-1]
		
	return results, position_m2m, position_f

def position_stop(prces, resid, num_std, ma_resid, std_resid):

	prces = np.asarray(prces)
	n = len(prces)
	position_f = np.zeros(n) 	# Final realized result 
	position_m2m = np.zeros(n) 	# Mark-to-market trading position
	results = np.zeros(n)
	# REMEMBER: resid = prediction - real_data
	#			if resid > 0 : Market is BEARISH and MARKET IS RIGHT (we follow the market feeling) 
	# 						  -> We are BEARISH and we go SHORT
	#			if resid < 0 : Market is BULLISH and MARKET IS RIGHT (we follow the market feeling)
	# 						  -> We are BULLISH and we go LONG
	for k in range(1,n-1):
	
		# OPENING POSITIONS -----------------------------------------------------

		
		if position_f[k-1] == 0 and resid[k] < ma_resid[k] - num_std * std_resid[k] \
		and resid[k] < 0 :
		
			position_f[k] =  prces[k] 					# Long position if resid < 0 
			position_m2m[k] = position_m2m[k-1]	
		
		elif position_f[k-1] == 0 and resid[k] > ma_resid[k] + num_std * std_resid[k] \
		and resid[k] < 0 :
		
			position_f[k] = - prces[k] 					# Short position if resid < 0 
			position_m2m[k] = position_m2m[k-1]
		
		elif position_f[k-1] == 0 and resid[k] > ma_resid[k] + num_std * std_resid[k] \
		and resid[k] > 0 :
		
			position_f[k] =  - prces[k] 					# Short position if resid > 0 
			position_m2m[k] = position_m2m[k-1]		
		
		elif position_f[k-1] == 0 and resid[k] < ma_resid[k] - num_std * std_resid[k] \
		and resid[k] > 0 :
		
			position_f[k] =  prces[k] 					# Long position if resid > 0 
			position_m2m[k] = position_m2m[k-1]	
		
		elif position_f[k-1] == 0 :
		
			position_m2m[k] = position_m2m[k-1]
			
		# WAITING ----------------------------------------------------------------------
		# WAITING in a SHORT position (from a BEARISH mkt <-> resid > 0)
		if position_f[k-1] < 0 and resid[k] > 0 and resid[k] > ma_resid[k] - num_std * std_resid[k]  :
			position_f[k] = position_f[k-1]				# Doing nothing w. Short position
			position_m2m[k] = position_m2m[k-1] + prces[k-1] - prces[k]	
		
		# WAITING in a SHORT position (from a BULLISH mkt <-> resid < 0)
		elif position_f[k-1] < 0 and resid[k] < 0 and resid[k] > ma_resid[k] - num_std * std_resid[k]  :
			position_f[k] = position_f[k-1]				# Doing nothing w. Short position
			position_m2m[k] = position_m2m[k-1] + prces[k-1] - prces[k]			
		
		# WAITING in a LONG position (from a BULLISH mkt <-> resid < 0)
		elif position_f[k-1] > 0 and resid[k] > 0 and resid[k] < ma_resid[k] + num_std * std_resid[k]  :
			position_f[k] = position_f[k-1]				# Doing nothing w. Long position
			position_m2m[k] = position_m2m[k-1] + prces[k] - prces[k-1]		
			
		# WAITING in a LONG position (from a BEARISH mkt <-> resid > 0)	
		elif position_f[k-1] > 0 and resid[k] < 0 and resid[k] < ma_resid[k] + num_std * std_resid[k]  :
			position_f[k] = position_f[k-1]				# Doing nothing w. Long position
			position_m2m[k] = position_m2m[k-1] + prces[k] - prces[k-1]					
			
		# CLOSING POSITIONS ----------------------------------------------------------------------------
		# Closing a SHORT position (from a BEARISH mkt <-> resid > 0)
		if position_f[k-1] < 0 and resid[k] < ma_resid[k] - num_std * std_resid[k]  :
			results[k] = - position_f[k-1] - prces[k] 	
			position_f[k] = 0
			position_m2m[k] = position_m2m[k-1] + prces[k-1] - prces[k]
			
		# CLOSING a SHORT position (from a BULLISH mkt <-> resid < 0)
		# elif position_f[k-1] < 0 and resid [k]<0 and resid[k] < ma_resid[k] - num_std * std_resid[k]  :
			# results[k] = - position_f[k-1] - prces[k] 	
			# position_f[k] = 0
			# position_m2m[k] = position_m2m[k-1] + prces[k-1] - prces[k]
		
		# Closing LONG position	(from a BULLISH mkt <-> resid < 0)
		elif position_f[k-1] > 0 and resid[k] > ma_resid[k] + num_std * std_resid[k]  :	
			results[k] = prces[k] - position_f[k-1] 	
			position_f[k] = 0
			position_m2m[k] = position_m2m[k-1] + prces[k] - prces[k-1]	
		
		# rst = results[0:k-1].cumsum()
		
		# PROFIT-TAKING / STOP-LOSS -----------------------------------------------		
		# Profit-Taking / Stop-Loss level for SHORT position (from a BEARISH mkt <-> resid > 0)
		rst = results[0:k-1].cumsum()
		if position_f[k-1] < 0 and (position_m2m[k-1]-rst[k-2] < -2 or position_m2m[k-1]-rst[k-2] > 2):
			results[k] = - position_f[k-1] - prces[k] 	# Closing Short-position
			position_f[k] = 0
			position_m2m[k] = position_m2m[k-1] + prces[k-1] - prces[k]
			
		# Profit-Taking / Stop-Loss level for LONG position	(from a BULLISH mkt <-> resid < 0)
		if position_f[k-1] > 0 and (position_m2m[k-1]-rst[k-2] < -2 or position_m2m[k-1]-rst[k-2] > 2) : 
			results[k] = prces[k] - position_f[k-1] 	# Closing Long-position
			position_f[k] = 0
			position_m2m[k] = position_m2m[k-1] + prces[k] - prces[k-1]	
		
		# ------------------------------------------------------------------------------------
		
	return results, position_m2m, position_f	
	
pr_year = prices.set_index(prices.index.year)

p_power = pr_year.Electricity
p_power = p_power.groupby(level = 0)

p_gas = pr_year.Gas
p_gas = p_gas.groupby(level = 0)

p_coal = pr_year.Coal
p_coal = p_coal.groupby(level = 0)

p_carbon = pr_year.Carbon
p_carbon = p_carbon.groupby(level = 0)
	
num_std = 1.75	

rslt09, pos_m2m09, pos_f09 = position(2009, p_power.values[2009], p_gas.values[2009], p_coal.values[2009],
 p_carbon.values[2009], res_09[:,0], num_std, ma_res_09, std_res09)	

rslt10, pos_m2m10, pos_f10 = position(2010, p_power.values[2010], p_gas.values[2010], p_coal.values[2010],
 p_carbon.values[2010], res_10[:,0], num_std, ma_res_10, std_res10)

rslt11, pos_m2m11, pos_f11 = position(2011, p_power.values[2011], p_gas.values[2011], p_coal.values[2011],
 p_carbon.values[2011], res_11[:,0], num_std, ma_res_11, std_res11)

rslt12, pos_m2m12, pos_f12 = position(2012, p_power.values[2012], p_gas.values[2012], p_coal.values[2012],
 p_carbon.values[2012], res_12[:,0], num_std, ma_res_12, std_res12)
 
rslt13, pos_m2m13, pos_f13 = position(2013, p_power.values[2013], p_gas.values[2013], p_coal.values[2013],
 p_carbon.values[2013], res_13[:,0], num_std, ma_res_13, std_res13)
 
 
# rslt09_s, pos_m2m09_s, pos_f09_s = position_stop(pr_year.values[2009], res_09[:,0], num_std, ma_res_09, std_res09)	
# rslt10_s, pos_m2m10_s, pos_f10_s = position_stop(pr_year.values[2010], res_10[:,0], num_std, ma_res_10, std_res10)
# rslt11_s, pos_m2m11_s, pos_f11_s = position_stop(pr_year.values[2011], res_11[:,0], num_std, ma_res_11, std_res11)
# rslt12_s, pos_m2m12_s, pos_f12_s = position_stop(pr_year.values[2012], res_12[:,0], num_std, ma_res_12, std_res12)

# ----------------------------------------------------------------------
# FIGURE 14 - 15 - 16 - 17 : Final Backtesting results without stop losses
# ----------------------------------------------------------------------
plt.rc('axes', grid=True)
plt.rc('grid', color='0.75', linestyle='-', linewidth=0.5)
fillcolor = 'darkgoldenrod'
textsize = 9
left, width = 0.1, 0.8
rect1 = [left, 0.4, width, 0.5] #left, bottom, width, height
rect2 = [left, 0.1, width, 0.3] #left, bottom, width, height
# rect3 = [left, 0.1, width, 0.2] #left, bottom, width, height

fig14 = plt.figure(facecolor='white')
axescolor  = '#f6f6f6'  # the axes background color

ax01 = fig14.add_axes(rect1, axisbg=axescolor) 
ax01.plot( ma_res_09,'k--', label='MA25_resid' )
ax01.plot( ma_res_09 + num_std*std_res09, 'r--', label='MA25+STD25_resid' )
ax01.plot( ma_res_09 - num_std*std_res09, 'r--', label='MA25-STD25_resid' )
ax01.plot( res_09[:,0], 'ro--', label='MA25_resid' )
ax01.legend(loc='best')
ax05 = ax01.twinx()
ax05.plot(p_power.values[2009], 'k-',lw=2, label='DEBY09')
ax05.legend(loc='best')
ax001 =  fig14.add_axes(rect2, axisbg=axescolor, sharex=ax01)
ax001.plot(rslt09[:-1].cumsum(), color='black', lw=2, label='ActualP&L09' )
ax001.plot(pos_m2m09[:-1], color='red', lw=1, label='M2M_Results09' )
ax001.legend(loc='best')
ax01.set_title('Results 2009')
ax001.axhline(0, color=fillcolor)
# ----------------------------------------------------------------------
fig15 = plt.figure(facecolor='white')
ax02 = fig15.add_axes(rect1, axisbg=axescolor) 
ax02.plot( ma_res_10, 'k--', label='MA25_resid' )
ax02.plot( ma_res_10 + num_std*std_res10, 'r--', label='MA25+STD25_resid' )
ax02.plot( ma_res_10 - num_std*std_res10, 'r--', label='MA25-STD25_resid' )
ax02.plot( res_10[:,0], 'ro--', label='MA25_resid' )
ax02.legend(loc='best')
ax02.set_title('Residuals2010')
ax05 = ax02.twinx()
ax05.plot(p_power.values[2010], 'k-',lw=2, label='DEBY10')
ax05.legend(loc='best')
ax002 =  fig15.add_axes(rect2, axisbg=axescolor, sharex=ax02)
ax002.plot(rslt10[:-1].cumsum(), color='black', lw=2, label='ActualP&L10'  )
ax002.plot(pos_m2m10[:-1], color='red', lw=1, label='M2M_Results10')
ax002.legend(loc='best')
ax002.set_title('Results 2010')
ax002.axhline(0, color=fillcolor)

# ----------------------------------------------------------------------
fig16 = plt.figure(facecolor='white')
ax02 = fig16.add_axes(rect1, axisbg=axescolor) 
ax02.plot( ma_res_11, 'k--', label='MA25_resid' )
ax02.plot( ma_res_11 + num_std*std_res11, 'r--', label='MA25+STD25_resid' )
ax02.plot( ma_res_11 - num_std*std_res11, 'r--', label='MA25-STD25_resid' )
ax02.plot( res_11[:,0], 'ro--', label='MA25_resid' )
ax02.legend(loc='best')
ax02.set_title('Residuals2011')
ax05 = ax02.twinx()
ax05.plot(p_power.values[2011], 'k-',lw=2, label='DEBY11')
ax05.legend(loc='best')
ax002 =  fig16.add_axes(rect2, axisbg=axescolor, sharex=ax02)
ax002.plot(rslt11[:-1].cumsum(), color='black', lw=2, label='ActualP&L11'  )
ax002.plot(pos_m2m11[:-1], color='red', lw=1, label='M2M_Results11')
ax002.legend(loc='best')
ax002.set_title('Results 2011')
ax002.axhline(0, color=fillcolor)
# ----------------------------------------------------------------------
fig17 = plt.figure(facecolor='white')
ax02 = fig17.add_axes(rect1, axisbg=axescolor) 
ax02.plot( ma_res_12, 'k--', label='MA25_resid' )
ax02.plot( ma_res_12 + num_std*std_res12, 'r--', label='MA25+STD25_resid' )
ax02.plot( ma_res_12 - num_std*std_res12, 'r--', label='MA25-STD25_resid' )
ax02.plot( res_12[:,0], 'ro--', label='MA25_resid' )
ax02.legend(loc='best')
ax02.set_title('Residuals2012')
ax05 = ax02.twinx()
ax05.plot(p_power.values[2012], 'k-',lw=2, label='DEBY12')
ax05.legend(loc='best')
ax002 =  fig17.add_axes(rect2, axisbg=axescolor, sharex=ax02)
ax002.plot(rslt12[:-1].cumsum(), color='black', lw=2, label='ActualP&L12'  )
ax002.plot(pos_m2m12[:-1], color='red', lw=1, label='M2M_Results12')
ax002.legend(loc='best')
ax002.set_title('Results 2012')
ax002.axhline(0, color=fillcolor)
# ----------------------------------------------------------------------
fig18 = plt.figure(facecolor='white')
ax02 = fig18.add_axes(rect1, axisbg=axescolor) 
ax02.plot( ma_res_13, 'k--', label='MA25_resid' )
ax02.plot( ma_res_13 + num_std*std_res13, 'r--', label='MA25+STD25_resid' )
ax02.plot( ma_res_13 - num_std*std_res13, 'r--', label='MA25-STD25_resid' )
ax02.plot( res_13[:,0], 'ro--', label='MA25_resid' )
ax02.legend(loc='best')
ax02.set_title('Residuals2013')
ax05 = ax02.twinx()
ax05.plot(p_power.values[2013], 'k-',lw=2, label='DEBY13')
ax05.legend(loc='best')
ax002 =  fig18.add_axes(rect2, axisbg=axescolor, sharex=ax02)
ax002.plot(rslt13[:-1].cumsum(), color='black', lw=2, label='ActualP&L13'  )
ax002.plot(pos_m2m13[:-1], color='red', lw=1, label='M2M_Results13')
ax002.legend(loc='best')
ax002.set_title('Results 2013')
ax002.axhline(0, color=fillcolor)

fig14.text(0.5,0.975,'QS1 - Yearly P&L without Stop-Losses and Profit-Taking levels', horizontalalignment='center',
verticalalignment='top')
fig15.text(0.5,0.975,'QS1 - Yearly P&L without Stop-Losses and Profit-Taking levels', horizontalalignment='center',
verticalalignment='top')
fig16.text(0.5,0.975,'QS1 - Yearly P&L without Stop-Losses and Profit-Taking levels', horizontalalignment='center',
verticalalignment='top')
fig17.text(0.5,0.975,'QS1 - Yearly P&L without Stop-Losses and Profit-Taking levels', horizontalalignment='center',
verticalalignment='top')
fig18.text(0.5,0.975,'QS1 - Yearly P&L without Stop-Losses and Profit-Taking levels', horizontalalignment='center',
verticalalignment='top')
show()