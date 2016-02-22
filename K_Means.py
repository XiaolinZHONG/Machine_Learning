#coding=utf-8
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import time
from scipy import sparse
import seaborn as sns
start_time = time.time()

data=np.loadtxt('d:\Pycharm\\3.txt')
'''
每一行代表一个好友关系。如第一行表示同学0与同学1的亲密程度为9（越高表示联系越密切）
'''
x_p=data[:,:2]
print x_p.T
y_p=data[:,2]
print y_p

x=(sparse.csc_matrix((y_p, x_p.T)).astype(float)).todense()
nUser=x.shape[1]
#x的行数就是用户编号，列x.shape[1],因为shape的返回值只有两个，一个行一个列
#print nUser
print x
print u'第1行：',x[0]
print u'第4行：',x[3]
plt.imshow(x,interpolation='nearest')
plt.xlabel('USER')
plt.ylabel('USER')
plt.xticks(range(nUser))
plt.yticks(range(nUser))
plt.show()

#K-Means
clf = KMeans(n_clusters=2,  n_init=1, verbose=1)
clf.fit(x)
print clf.cluster_centers_   #打印相应的分类中心
print(clf.labels_)#每个样本所属的簇 这里是以行来分类的！！！
print clf.inertia_
#用来评估簇的个数是否合适，距离越小说明簇分的越好，选取临界点的簇个数
'''
#进行预测
print clf.predict(feature)

#保存模型
joblib.dump(clf , 'c:/km.pkl')

#载入保存的模型
clf = joblib.load('c:/km.pkl')
'''
#K-Means
clf = KMeans(n_clusters=3,  n_init=1, verbose=1)
#n_cluster表示的是区分的聚类的数目，后面的n_init表示的是显详细的
clf.fit(x)
print clf.cluster_centers_
print(clf.labels_)
print clf.inertia_
#K-Means
clf = KMeans(n_clusters=2,  n_init=1, verbose=1)
#这里的verbose的值越高表示显示的信息的详细
clf.fit(x)
print clf.cluster_centers_
print(clf.labels_)

#K-Means
print np.vstack([x[0],x[2],x[3]])
#选取这三个点当做分类的初始点
clf=KMeans(n_clusters=3,init=np.vstack([x[0],x[2],x[1]]),verbose=1)
clf.fit(x)
print clf.cluster_centers_
print clf.labels_

#K-Means
clf=KMeans(n_clusters=3,init='random',verbose=1)
clf.fit(x)
print clf.cluster_centers_
print clf.labels_

#K-Means
clf=KMeans(n_clusters=3,init='k-means++',verbose=1)
clf.fit(x)
print clf.cluster_centers_
print clf.labels_