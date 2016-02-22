#coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.stats import norm
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

#生成数据
x=np.arange(0,1,0.002)#生成0到1的步长为0.02的列表
y=norm.rvs(0,size=500,scale=0.1)#表示以0位均衡点，不超过0.2（可以是0.1**）产生随机数500个
#产生随机数的方法有很多，这只是其中一个
#print x,y
y=y+x**2
#print y

#均方误差 RMSE是预测值与真实值的误差平方根的均值
def rmse(y_test,y_true):
    return sp.sqrt(sp.mean((y_test - y_true) ** 2))
#R2方法是将预测值跟只使用均值的情况下相比，看能好多少。其区间通常在（0,1）之间。
# 0表示还不如什么都不预测，直接取均值的情况，而1表示所有预测跟真实结果完美匹配的情况。
#R2的计算方法，不同的文献稍微有不同。
def R2(y_test, y_true):
    return 1 - ((y_test - y_true)**2).sum() / ((y_true - y_true.mean())**2).sum()
#R22函数的实现来自Conway的著作《机器学习使用案例解析》，不同在于他用的是2个RMSE的比值来计算R2
def R22(y_test,y_true):
    y_mean=np.array(y_true)
    y_mean[:]=y_mean.mean()
    return 1-rmse(y_test,y_true)/rmse(y_mean,y_true)
plt.scatter(x,y,s=2)#后面的S表示点的大小
degree=[1,2,100]#生成一个列表，包含3个数，分别是1,2,100，因为后面还要分这三种情况拟合
y_test=[]#生成一个空的列表
y_test=np.array(y_test)#把这个列表变成数组
#只能使用上面的方法，→_→的方法不行，y_test=np.array()

for d in degree:
    clf=Pipeline([('poly',PolynomialFeatures(degree=d)),('linear',LinearRegression(fit_intercept=False))])#fit_intercept拟合窃听/拦截？？？
    #上面的方法就是回归分析，其中前面的部分就是表示分别拟合1,2,100这三种情况，后面表示使用的是线性拟合？？？
    '''
    #clf=Pipeline([('poly',PolynomialFeatures(degree=d)),('linear',linear_model.Ridge())])
    #polynomial是多项式的意思，后面的degree表示的是
    #clf=Pipeline([('poly',PolynomialFeatures(degree=d)),('linear',linear_model.Ridge())])
    #和上面相同表示分别拟合回归三种情况，后面部分表示的是选择的回归模型是岭回归（即带有惩罚机制的回归分析）
    #clf=Pipeline([('poly',PolynomialFeatures(degree=d)),('linear',linear_model.LogisticRegression())])
    #linear_model.后面可以跟随很多回归模型,例如上面就是逻辑回归模型
    #clf=Pipeline([('poly',PolynomialFeatures(degree=d)),('linear',linear_model.Lasso())])
    #套锁回归
    #clf=Pipeline([('poly',PolynomialFeatures(degree=d)),('linear',linear_model.ElasticNet())])
    #ElasticNet是套锁回归和领回归的混合
    '''
    clf.fit(x[:400,np.newaxis],y[:400])#这一行表示用来真正拟合的
    y_test=clf.predict(x[:,np.newaxis])#这一行是用来做预测的
    print clf.named_steps['linear'].coef_#得到相应的各个系数的值，分别是从最低次开始
    #类似于前面的
    print 'rmse=%.2f, R2=%.2f, R22=%.2f, clf.score=%.2f' % \
          (rmse(y_test, y),#评估性能
           R2(y_test, y),#评估性能
           R22(y_test, y),#评估性能
           clf.score(x[:, np.newaxis], y))
    plt.plot(x,y_test,linewidth=2)#显示预测得到的曲线
plt.grid()#显示网格
plt.legend(['1','2','100'],loc='upper left')#显示标签
plt.show()
'''
做回归分析，常用的误差主要有均方误差根（RMSE）和R-平方（R2）。

RMSE是预测值与真实值的误差平方根的均值。这种度量方法很流行（Netflix机器学习比赛的评价方法），是一种定量的权衡方法。

R2方法是将预测值跟只使用均值的情况下相比，看能好多少。其区间通常在（0,1）之间。0表示还不如什么都不预测，直接取均值的情况，而1表示所有预测跟真实结果完美匹配的情况。

R2的计算方法，不同的文献稍微有不同。如本文中函数R2是依据scikit-learn官网文档实现的，跟clf.score函数结果一致。

而R22函数的实现来自Conway的著作《机器学习使用案例解析》，不同在于他用的是2个RMSE的比值来计算R2。

我们看到多项式次数为1的时候，虽然拟合的不太好，R2也能达到0.82。2次多项式提高到了0.88。而次数提高到100次，R2也只提高到了0.89。
'''