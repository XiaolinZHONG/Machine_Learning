#coding=utf-8
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from scipy import stats as ss
import seaborn as sns
import numpy as np

csvfile=file('d:\\test1229.csv','rb')
df=pd.read_csv(csvfile)
print u'全部的ONT设备数：%s'%df.index
for i in range(1,13):
    print df.ix[:,i].describe()

x=np.arange(1,1079)
y=df.ix[0:1077,1]
plt.plot(x,y,'k.')

print df.ix[1:,3].head(6)
print df.ix[:,3].head(6)
print df.index
print df.columns
df.T#转置
#print df.drop(df.columns[[2,4]],axis=1)
#上面的语句表示舍弃第二列和第四列，若axis=0则表示舍弃第二行和第四行
#print ss.ttest_1samp(a=df.ix[:10,3],popmean=-23)
#假设性检验 在进行该操作之前应该首先进行数据的清洗。
#pd.options.display.mpl_style='default'
plt.show()