#coding=utf-8
import numpy as np

data=[]
labels=[]
with open('d:\Pycharm\\1.txt',) as ifile:
      for line in ifile:
            tokens=line.strip().split(' ')#去掉空格
            data.append([float(tk) for tk in tokens[:-1]])
            labels.append(tokens[-1])
x=np.array(data)
labels=np.array(labels)
y=np.zeros(labels.shape)
print x

y[labels=='fat']=1

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.0)

'''creat a mesh to plot in'''
x_min,x_max=x_train[:,0].min()-0.1,x_train[:,0].max()+0.1

y_min,y_max=x_train[:,1].min()-1,x_train[:,1].max()+1

h=0.02
xx,yy=np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))


'''SVM'''
titles=['linearSVC(linear kernel)',
        'SVC with polynomial(degree 3) kernel',
        'SVC with RBF Kernel','SVC with SIgmiod kernel']
clf_linear=svm.SVC(kernel='linear').fit(x,y)

clf_poly=svm.SVC(kernel='poly',degree=3).fit(x,y)

clf_rbf=svm.SVC().fit(x,y)

clf_sigmoid=svm.SVC(kernel='sigmoid').fit(x,y)

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
