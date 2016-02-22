#coding=utf-8
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab
from sklearn.cross_validation import train_test_split
import pandas as pd
import time
from scipy import sparse
import seaborn as sns
start_time = time.time()

data=np.loadtxt('d:\Pycharm\\2.txt')
#print data


#计算向量test与data数据每一个向量的相关系数，data一行为一个向量
def calc_relation(testfor, data):
    return np.array(
        [np.corrcoef(testfor, c)[0,1]
         for c in data])
'''
这里有很多需要讲解的地方
'''

#实际上就是皮尔森系数的计算公式
def all_correlations(y, X):
    X = np.asanyarray(X, float)
    y = np.asanyarray(y, float)
    xy = np.dot(X, y)
    y_ = y.mean()
    ys_ = y.std()
    x_ = X.mean(1)
    xs_ = X.std(1)
    n = float(len(y))
    ys_ += 1e-5  # Handle zeros in ys
    xs_ += 1e-5  # Handle zeros in x
    return (xy - x_ * y_ * n) / n / xs_ / ys_


#数据读入

x_p = data[:, :2] # 取前2列，购买人，购买物品
y_p = data[:,  2] # 取前2列购买的数量
x_p -= 1          # 0为起始索引,??因为后面我们还要使用x_p的值来表示行号和列号，行列号都是以0开始的

y = (sparse.csc_matrix((y_p, x_p.T)).astype(float)).todense()
'''
生成一个矩阵：sparse.csc_matrix(('数据', '这个部分是一个两维的，x_p.T的值分别表示行和列')).astype(float)).todense()
最后的语句的意思是产生稠密矩阵，还可以使用:.toarray()
这个矩阵的横纵坐标分别表示的是：购买者和他购买的东西，颜色的深浅表示的是购买的数量的多少。
实际上这个函数本来的意义是把这个矩阵分别表示成三列数据来存储矩阵，来节省空间，这里是反向应用了.
对于推荐用户购买物品，一个用户可能购买很多个商品，有的用户可能没有购买很多个商品，那么对应的商品的数目为0，当
0的数目比较多的时候就是所谓的稀疏矩阵。同样的对于用户的喜欢某些新闻也可以使用类似的算法。
'''

nUser, nItem = y.shape
print y.shape
print x_p,'\n',y_p,y


#可视化矩阵
plt.imshow(y,interpolation='nearest')
plt.xlabel('Item')
plt.ylabel('user')
plt.xticks(range(nItem))
plt.yticks(range(nUser)) #表示y轴的坐标值范围
plt.show()

#加载数据集，实际上并没有切分数据集
x_p_train, x_p_test, y_p_train, y_p_test = \
          train_test_split(data[:,:2], data[:,2], test_size = 0.0)
x = (sparse.csc_matrix((y_p_train, x_p_train.T)).astype(float)).todense()
print x.T
#这里把矩阵转置，是因为后面的计算都是基于用户的，以物品作为属性（特征）来计算相似度，
# 特征类似于向量的基矢，通过计算余弦来得到相似度

Item_likeness = np.zeros((nItem, nItem))
#创造一个用来存储相关系数的空矩阵，显然这个矩阵的大小是以用户数来衡量的

#训练
#print Item_likeness
for i in range(nItem):
    Item_likeness[i] =np.array([np.corrcoef(x[:,i].T, c)[0,1] for c in x.T])
    #print u'这是第列x','\n',x[:,i],'\n'
    print u'这是第i列x.T:','\n',x[:,i].T
    X1=np.corrcoef(x[:,i].T, x.T[0])
    print '\n',u'这里是相关性计算','\n',X1
    X2=np.corrcoef(x[:,i].T, x.T[1])
    print '\n',u'这里是相关性计算','\n',X2
    X3=np.corrcoef(x[:,i].T, x.T[2])
    print '\n',u'这里是相关性计算','\n',X3
    X4=np.corrcoef(x[:,i].T, x.T[3])
    print '\n',u'这里是相关性计算','\n',X4
    X5=np.corrcoef(x[:,i].T, x.T[4])
    print '\n',u'这里是相关性计算','\n',X5
    XX=np.array([X1,X2,X3,X4,X5])
    print '\n',u'相关性矩阵','\n',XX
    #print '\n',u'这里是相关性计算','\n',np.corrcoef(x[:,i].T, c)
    #print '\n',u'这是完整的 x.T','\n',x.T
    Item_likeness[i,i] = -1
    #这里减1是因为：自身和自身的相关性为1，应该去掉
    print 'THIS ISCORR COV',Item_likeness[i]

for t in range(Item_likeness.shape[1]):
    item = Item_likeness[t].argsort()[-3:]
    '''
    取相关系数的排名的前三名，并获取相应的列数。这里转置后的矩阵的列数对应的就是商品
    '''
    print("Buy Item %d will buy item %d,%d,%d "%
          (t, item[0], item[1], item[2]))

print("time spent:", time.time() - start_time)

'''
______________下面使用seaborn 中的函数绘制相关性曲图____________
这个和前面绘制的相关性图，可以利用前面的
'''
data=np.loadtxt('d:\Pycharm\\2.txt')
#data[:,1:3]实际上显示的是第二和第三列（注意所有的列数是从0开始的）
df1=pd.DataFrame(data[:,1:3],index=data[:,0],columns=list(['item','times']))
df1.index.name='user'
#print df1
COL=list(['user1','user2','user3','user4','user5','user6','user7'])
df2=pd.DataFrame(x.T,columns=COL)
print df2

sns.set(style='darkgrid')
style='white'

# Compute the correlation matrix
#计算相关性
corr = df2.corr()
#类似于皮尔森系数，是一个协方差除以方差的函数

# Generate a mask for the upper triangle
#给上面的相关性计算结果生成相应的码
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
#设置图片的大小
f, ax = plt.subplots(figsize=(7, 5))
#print f,'\n',ax

# Generate a custom diverging colormap
#这里是创造一个多色彩的彩色盘
cmap = sns.diverging_palette(200, 10,as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
#生成热力图
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3,linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
# xticklabels=2, yticklabels=2,设置坐标轴显示的
# square=False ?不知何用
#cbar_kws={"shrink": .5} 显示旁边的color bar即旁边的程度表

plt.show()
