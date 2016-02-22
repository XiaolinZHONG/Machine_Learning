#coding=utf-8
import scipy as sp
import numpy as np
from sklearn.naive_bayes import MultinomialNB
#这里选取的贝叶斯选择器是使用的出现的次数作为特征值进行选择
'''
from sklearn.naive_bayes import GaussianNB
#这里选取的贝叶斯的选择器是使用高斯分布的特征值
from sklearn.naive_bayes import BernoulliNB
#这里选取的是贝叶斯的白努力分布（二值分布）
'''
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
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
print x
print '\n',y
print '\n',labels
#T to 01
y[labels=='fat']=1 #把fat的标记为1，其余的不变化，即还是0
# 拆分两组
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
#选取整个数据的20%作为测试用，前面的8层作为训练集,这种选择是随机的
print '\n',x_train
print y_train
print x_test
print y_test

##############################
clf=MultinomialNB()
clf.fit(x_train,y_train)
doc_class_predicted=clf.predict(x_test)
##############################

print (np.mean(doc_class_predicted==y_test))
precision,recall,thresholds=precision_recall_curve(y_test,clf.predict(x_test))
answer=clf.predict_proba(x_test)[:,1]
report=answer>0.5
print classification_report(y_test,report,target_names=['fat','thin'])
