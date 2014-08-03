from pandas import Series, DataFrame,isnull, ExcelFile, date_range
from pylab import plot, show
import pandas as pd
import pylab as pl
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
# cls
path_xls = 'C:\Users\Suso\Desktop\Ipython Notebooks\Naivereg2.csv'
xls_data = pd.read_csv(path_xls,parse_dates=True, index_col=0)
prices = xls_data.dropna() # we have now 1240 prices from 2008

commdty_corr = lambda x: x.corrwith(x['Electricity'])
    
by_year = prices.groupby(lambda x: x.year)
by_month = prices.groupby(lambda x: x.month)

print 'Correlation Table per year'
print '='*80
print by_year.apply(commdty_corr)
print '='*80
# by_month.apply(commdty_corr).plot()

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
	
def predict_in(data,yvar,xvars): # in-sample prediction
    Y = data[yvar]
    X = data[xvars]
    X['intercept'] = 1.
    #olsmod = sm.OLS(Y,X)
    #olsres = olsmod.fit()
    predicti2 = sm.OLS(Y,X).fit()
    predict_in = predicti2.predict(X)
    
    #print result.summary
    #print result.summary
    return predict_in

	
results1 = by_year.apply(regress, 'Electricity', ['Gas','Coal','Carbon'])
results2 = by_year.apply(residual, 'Electricity', ['Gas','Coal','Carbon'])

# Ploting of the histograms of residuals 
# --------------------------------------------------
plt.close('all')
plt.rc('axes',grid=True)
plt.rc('grid',color='0.75',linestyle='-',linewidth=0.5)
fig = plt.figure()
ax1 = fig.add_subplot(3,2,1)
ax2 = fig.add_subplot(3,2,2)
ax3 = fig.add_subplot(3,2,3)
ax4 = fig.add_subplot(3,2,4)
ax5 = fig.add_subplot(3,2,5)
ax6 = fig.add_subplot(3,2,6)

ax1.hist(results2[2008],bins=25, alpha = 0.2, color='g', normed=True)
ax1.set_title('2008')
ax2.hist(results2[2009],bins=25, alpha = 0.2, color='c', normed=True)
ax2.set_title('2009')
ax3.hist(results2[2010],bins=25, alpha = 0.2, color='b', normed=True)
ax3.set_title('2010')
ax4.hist(results2[2011],bins=25, alpha = 0.2, color='k', normed=True)
ax4.set_title('2011')
ax5.hist(results2[2012],bins=25, alpha = 0.2, color='r', normed=True)
ax5.set_title('2012')
ax6.hist(results2[2013],bins=25, alpha = 0.2, color='y', normed=True)
ax6.set_title('2013')

fig.text(0.5,0.975,'OLS-Fuel Regress: Histogram of residuals in-the-sample', horizontalalignment='center',
verticalalignment='top')

# Calcualtion of the prediction out-of-sample

data2 = prices.set_index(prices.index.year)
# m = data2.ix[2009:].electricity.count()
# prediction = DataFrame(np.zeros(m),index=prices.index[-m:])
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
prediction = nn[data2.Electricity.count()-results3.shape:]
real_data = data2.Electricity.values[data2.Electricity.count()-results3.shape:]
residual_out = prediction - real_data
residual_ots = DataFrame(residual_out, index = prices.index[data2.Electricity.count()-results3.shape:].year)

# by_year_residual_ots = residual_ots.groupby(lambda x: x.year)



# plotting out-of-the-sample results
fig1 = plt.figure()
ax1 = fig1.add_subplot(2,1,1)
ax1.plot(real_data)
ax1.plot(prediction)
ax2 = fig1.add_subplot(2,1,2)
ax2.hist(residual_ots.ix[2009].values,bins=25, alpha = 0.2, color='c', normed=True)
ax2.hist(residual_ots.ix[2010].values,bins=25, alpha = 0.2, color='b', normed=True)
ax2.hist(residual_ots.ix[2011].values,bins=25, alpha = 0.2, color='k', normed=True)
ax2.hist(residual_ots.ix[2012].values,bins=25, alpha = 0.2, color='r', normed=True)
# ax2.hist(residual_out[2:], bins=25)


fig1.text(0.5,0.975,'Actual vs Predictions out-the-sample', horizontalalignment='center',
verticalalignment='top')

props = font_manager.FontProperties(size=10)
leg = ax1.legend(loc='center right', shadow=True, fancybox=True, prop=props) 
# leg.get_frame().set_alpha(0.5)


fig2 = plt.figure()
ax1 = fig2.add_subplot(2,2,1)
ax2 = fig2.add_subplot(2,2,2)
ax3 = fig2.add_subplot(2,2,3)
ax4 = fig2.add_subplot(2,2,4)
ax1.hist(residual_ots.ix[2009].values,bins=25, alpha = 0.2, color='c', normed=True)
ax1.set_title('2009')
ax2.hist(residual_ots.ix[2010].values,bins=25, alpha = 0.2, color='b', normed=True)
ax2.set_title('2010')
ax3.hist(residual_ots.ix[2011].values,bins=25, alpha = 0.2, color='k', normed=True)
ax3.set_title('2011')
ax4.hist(residual_ots.ix[2012].values,bins=25, alpha = 0.2, color='r', normed=True)
ax4.set_title('2012')
fig2.text(0.5,0.975,'OLS-Fuel Regress: Histogram of residuals out-the-sample', horizontalalignment='center',
verticalalignment='top')

# results3 = by_year.apply(predict_in, 'Electricity', ['Gas','Coal','Carbon'])

print ' ' 
print 'Coefficientes of the Regression ' 
print '-'*80
print results1 #['parameters']
print '-'*80


show()
