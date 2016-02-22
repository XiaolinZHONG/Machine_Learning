#coding=utf-8
#注意现在还是没有解决中显示乱码的状况
import pandas as pd
#from scipy import stats as ss

import matplotlib.pyplot as plt

csvfile=file('d:\Pycharm\\forpython.csv','rb')

df=pd.read_csv(csvfile)

print df.head()
print df.columns
print df.index
print df.ix[:].head(4)
print df.describe()
#print ss.ttest_1samp(a=df.ix[:,3],popmean=14000)

plt.show(df.plot(kind='box'))
#? pd.options.display.mpl_style='defeaut'
#? df.plot(kind='box')
