#coding=utf-8
################
##决策树
################
##我犯的错误
##测试用的数据的TXT文档中第一行是空值
################
import numpy as np
import scipy as sp
from sklearn import tree
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split #是用来分离训练和测试数组的
#inputdata
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
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)#选取整个数据的20%作为测试用，前面的8层作为训练集,这种选择是随机的
print '\n',x_train
print y_train
print x_test
print y_test
#使用信息熵作为划分标准
clf=tree.DecisionTreeClassifier(criterion='entropy')#使用信息熵作为划分标准

print clf
clf.fit(x_train,y_train)
with open('d:\Pycharm\\tree.dot','w') as f:
      f=tree.export_graphviz(clf,out_file=f)

print (clf.feature_importances_)#显示每一个特征的影响力，很显然在这个例子中体重的影响力更大
answer=clf.predict(x_train)
print '************'
print x_train
print answer
print y_train
print (np.mean(answer==y_train))
precision,recall,thresholds=precision_recall_curve(y_train,clf.predict(x_train))
answer=clf.predict_proba(x)[:,-1]
print (classification_report(y,answer,target_names=['thin','fat']))

