#coding=utf-8
##################
#KNN test
#author_Xiaolin
##################
import numpy as np
from sklearn import neighbors
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
#上面的库是交互验证的库
import matplotlib.pyplot as plt

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
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
#选取整个数据的20%作为测试用，前面的8层作为训练集,这种选择是随机的

##创建网格##
h=0.01
x_min,x_max=x[:1].min()-0.1,x[:1].max()+0.1
#print x[:1].min()
#x坐标表示的是第一列的身高
y_min,y_max=x[:2].min()-0.1,x[:2].max()+0.01
#y坐标表示的是第二列的体重
xx,yy=np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
#上面是一个创建网格的一个典型教案


##创建分类信息KNN
clf=neighbors.KNeighborsClassifier(algorithm='kd_tree')
'''
KNeighborsClassifier可以设置3种算法：‘brute’，‘kd_tree’，‘ball_tree’。
# 如果不知道用哪个好，设置‘auto’让KNeighborsClassifier自己根据输入去决定。
#clf=neighbors.BallTree()当维度超过20后使用balltree???????
'''
clf.fit(x_train,y_train)
answer=clf.predict(x)
#注意这里使用的不是x_test因为是要对整个数据进行区分

print x
print answer
print y
print np.mean(answer==y)#计算answer值和Y值相同的占比为多少

##准确率
precision,recall,thresholds=precision_recall_curve(y_train,clf.predict(x_train))
answer=clf.predict_proba(x)[:-1]
#print answer
print (classification_report(y,answer,target_names=['thin','fat']))

#answer=clf.predict_proba(np.c_[xx.ravel(),yy.ravel()])[:,1]
#z=answer.reshape(xx.shape)
#plt.contourf(xx,yy,z,cmap=plt.cm.Paired,alpha=0.8)


#plt.scatter(x_train[:,1], x_train[:,2], c=y_train, cmap=plt.cm.Paired)
#plt.xlabel(u'身高')
#plt.ylabel(u'体重')
#plt.show()