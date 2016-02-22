#coding=utf-8
from sklearn import datasets
from sklearn import svm


#################
iris=datasets.load_iris()
print iris.data
print iris.data.shape
print iris.target
##################
clf=svm.SVC(gamma=0.01,C=100.0)#?SVM向量机
print clf
##################
#下面这个程序实际上是一个图像识别程序
digits=datasets.load_digits()# 引入需要测试的数据 实际上是一个图片对应数值得表
print digits
clf.fit(digits.data[:-5],digits.target[:-5])#进行相应的拟合，首先选取前n-1个数据的X值以及其对应的Y值来作为训练集
result=clf.predict(digits.data[-2])#输入第N 个X值，来预测相应的Y值。
print result


#下面使用手动显示来验证数值
import matplotlib.pyplot as plot
#plot.figure(1,figsize=(3,3))#这里的1表示第一幅画图， 后面表示的是图片的大小
plot.figure(2,figsize=(6,6))#上面的情况类似，但是表示的是第二幅图，当有两幅图的时候，实际上显示的是最后一幅图，除非每一个后面都跟一个相应的
plot.imshow(digits.images[-2],cmap=plot.cm.gray_r,interpolation='nearest')
plot.show()
###################

import matplotlib.pyplot as plot
from sklearn import datasets,svm,metrics
digits=datasets.load_digits()
imges_and_labels=list(zip(digits.images,digits.target))
for index, (imge,label) in enumerate(imges_and_labels[:4]):
    plt.subplot(2,4,index+1)
    plt.axis('off')
    plt.imshow(image,cmap=plt.cm)