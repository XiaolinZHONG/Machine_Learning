#coding=utf-8
import numpy as np
import scipy as sp
from sklearn import svm
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

data=[]
labels=[]
with open('d:\Pycharm\\1.txt',) as ifile:
      for line in ifile:
            tokens=line.strip().split(' ')#去掉空格
            data.append([float(tk) for tk in tokens[:-1]])
            #一直到倒数第一个，其中不包括倒数第一个,也可以是单独的值
            #data.append([float(tk) for tk in tokens[-3:-1]]
            # #表示从倒数第三个一直数到倒数第一个不包括倒数第一
            labels.append(tokens[-1])
x=np.array(data)
labels=np.array(labels)
y=np.zeros(labels.shape)
print x
#print '\n',y
print '\n',labels,'\n\n\n'
#T to 01
y[labels=='fat']=1
#把fat的标记为1，其余的不变化，即还是0
# 拆分两组
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.0)

'''creat a mesh to plot in'''
x_min,x_max=x_train[:,0].min()-0.1,x_train[:,0].max()+0.1
#获取x_train值中的第一列（即身高），并从中选取最小值和最大值
y_min,y_max=x_train[:,1].min()-1,x_train[:,1].max()+1
#获取x_train值中的第二列（即体重），并从中选取最小值和最大值

h=0.02
xx,yy=np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
'''
这一句是生成一个网格，网格的横坐标是np.arange(x_min,x_max,h)，纵坐标是np.arange(y_min,y_max,h),这样实际上是生成一个
(70, 40)
[[ 1.   1.1  1.2 ...,  4.7  4.8  4.9]
 [ 1.   1.1  1.2 ...,  4.7  4.8  4.9]
 [ 1.   1.1  1.2 ...,  4.7  4.8  4.9]
 ...,
 [ 1.   1.1  1.2 ...,  4.7  4.8  4.9]
 [ 1.   1.1  1.2 ...,  4.7  4.8  4.9]
 [ 1.   1.1  1.2 ...,  4.7  4.8  4.9]]
(70, 40)
[[20 20 20 ..., 20 20 20]
 [21 21 21 ..., 21 21 21]
 [22 22 22 ..., 22 22 22]
 ...,
 [87 87 87 ..., 87 87 87]
 [88 88 88 ..., 88 88 88]
 [89 89 89 ..., 89 89 89]]
这个样子的数组
'''

'''SVM'''
titles=['linearSVC(linear kernel)',
        'SVC with polynomial(degree 3) kernel',
        'SVC with RBF Kernel','SVC with SIgmiod kernel']
clf_linear=svm.SVC(kernel='linear').fit(x,y)
#clf_linear=svm.linearSVC().fit()
#线性函数
clf_poly=svm.SVC(kernel='poly',degree=3).fit(x,y)
#多项式函数 3阶
clf_rbf=svm.SVC().fit(x,y)
#径向函数
clf_sigmoid=svm.SVC(kernel='sigmoid').fit(x,y)
#sigmoid 函数
#xx=np.arange(x_min,x_max,h)
#yy=np.arange(y_min,y_max,h)
#print xx.ravel()
#print yy.ravel()
for i,clf in enumerate((clf_linear, clf_poly, clf_rbf, clf_sigmoid)):
    '''同时调用函数和统计数目使用，后面同时跟一个例子
    for idx,clor in enumerate('rgbck')'''
    answer=clf.predict(x_train)
    print clf
    print np.mean(answer==y_train)
    #正常情况下应该是answer==y_test，本文并没有28分
    print answer
    print y_train

    plt.subplot(2,2,i+1) #i
    #显示两行两列的字图片，其中
    plt.subplots_adjust(wspace=0.4,hspace=0.4)
    #设置字图片的高和宽参数
    # Put the result into a color plot
    #answer=clf.predict_proba(np.c_[xx.ravel(),yy.ravel()])
    probality=np.c_[xx.ravel(),yy.ravel()]
    '''
    (2800,) [ 1.   1.1  1.2 ...,  4.7  4.8  4.9] 注意这里是2800行1列
    (2800,) [20 20 20 ..., 89 89 89]
    很显然实际上是在压扁
    hstack函数的功能，参见numpytest
    [  1.    1.1   1.2 ...,  89.   89.   89. ]
    np.c_de 的功能
    [[  1.   20. ]
    [  1.1  20. ]
    [  1.2  20. ]
    ...,
    [  4.7  89. ]
    [  4.8  89. ]
    [  4.9  89. ]]
    '''
    #print probality
    answer=clf.predict_proba(probality)
    '''np.c_[np.array([[1,2,3]]), 0, 0, np.array([[4,5,6]])]结果就是
    array([[1, 2, 3, 0, 0, 4, 5, 6]])，显示实际上是做数据拼接.因为训练数据是两维的
    其中ravel是把一个N*M的变成1*NM的
    '''
    print xx.shape
    print answer
    z=answer.reshape(xx.shape)
    plt.contourf(xx, yy,z, cmap=plt.cm.Paired, alpha=0.8)
    #轮廓线跟踪contour following
    '''
    这里是再绘画一个二维的xy函数，但是是以z来表征颜色的深浅
    '''
    # Plot also the training points
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=plt.cm.Paired)
    '''散点图，求中C表示的是颜色，很显然这里是根据y_train来判断颜色
    后面的cmap只有在前面的C 是一个数组的时候使用'''
    plt.xlabel(u'height')
    plt.ylabel(u'weight')
    #plt.xlim(xx.min(), xx.max())
    #plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

plt.show()
