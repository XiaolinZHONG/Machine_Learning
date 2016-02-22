#coding=utf-8
#向量机
from sklearn import svm
import numpy as np

from sklearn.cross_validation import train_test_split
clf=svm.SVC(gamma=0.5,C=10,kernel='rbf')
#其中核函数中的gamma函数设置(针对多项式/rbf/sigmoid核函数) (默认类别数目的倒数)
#C一般可以选择为：10^t , t=- 4..4就是0.0001 到10000，选择的越大，表示对错误例惩罚程度越大，可能会导致模型过拟合
#kernel 表示内核是什么函数
data=[]
labels=[]
with open('d:\Pycharm\\1.txt',) as ifile:
      for line in ifile:
            tokens=line.strip().split(' ')#去掉空格
            data.append([float(tk) for tk in tokens[:-1]])#一直到倒数第一个，其中不包括倒数第一个,也可以是单独的值
            #data.append([float(tk) for tk in tokens[-3:-1]]#表示从倒数第三个一直数到倒数第一个不包括倒数第
            labels.append(tokens[-1])
x=np.array(data)
labels=np.array(labels)
y=np.zeros(labels.shape)
y[labels=='fat']=1 #把fat的标记为1，其余的不变化，即还是0
# 拆分两组
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)#选取整个数据的20%作为测试用，前面的8层作为训练集,这种选择是随机的
clf.fit(x_train,y_train)
answer=clf.predict(x_train)
print '************'
print x_train
print answer
print y_train
print (np.mean(answer==y_train))
#print (clf.feature_importances_)向量机没有这项指标
#通过这个判断一个人是不是胖子的程序我们发现，向量机的准确率并不是很高
#所以说每一个算法都有他自己擅长的地方
