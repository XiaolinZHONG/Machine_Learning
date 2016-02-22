#coding=utf-8
########################
######情感分类，贝叶斯可以达到比较高的水平
########################
from sklearn.datasets import load_files
import scipy as sp
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report

movie_reviews=load_files('d:\Pycharm\\tokens')
'''
上面的数据是通过load files来获取已经存放在本地的数据文件
通过save口令将数据保存起来不用每次去读取硬盘
'''
#save
sp.save('movie_data.npy',movie_reviews.data)
sp.save('movie_target.npy',movie_reviews.target)
#read
movie_data=sp.load('movie_data.npy')
movie_target=sp.load('movie_target.npy')

#BOOL
######################################################################################
#####IT-TDF 文本数据矢量化
######################################################################################

'''
TF（词频）的计算很简单，就是针对一个文件t，某个单词Nt 出现在该文档中的频率。
比如文档“I love this movie”，单词“love”的TF为1/4。如果去掉停用词“I"和”it“，则为1/2。
IDF（逆向文件频率）的意义是，对于某个单词t，可以由总文件数目除以包含该词语之文件的数目，再将得到的商取对数得到。
IDF是为了凸显那种出现的少，但是占有强烈感情色彩的词语。
比如“movie”这样的词的IDF=ln(12/5)=0.88，远小于“love”的IDF=ln(12/1)=2.48。
stop_words = 'english'，表示使用默认的英文停用词。可以使用count_vec.get_stop_words()查看TfidfVectorizer内置的所有停用词。
当然，在这里可以传递你自己的停用词list（比如这里的“movie”）
注意这些数据集可能存在非法字符问题。所以在构造count_vec时，传入了decode_error = 'ignore'，以忽略这些非法字符。
count_vec构造时默认传递了max_df=1，因此TF-IDF都做了规格化处理，以便将所有值约束在[0,1]之间。
'''
count_vec=TfidfVectorizer(binary=False,decode_error='ignore',stop_words='english')
x_train, x_test, y_train, y_test\
    = train_test_split(movie_data, movie_target, test_size = 0.2)

x_train=count_vec.fit_transform(x_train)
x_test=count_vec.transform(x_test)
x=count_vec.transform(movie_data)
y=movie_target
'''
下面是个人理解：
矢量化的精髓在于使用的是transform函数，首先把前面2 8分的数据中8成用于训练文本的矢量化，即本文中的x_train
其中count_vec.fit_transform（注意这里面只有所谓的x_train）把文本制作成一个维度非常大的稀疏矩阵，并计算出相应的每一个结果对应的相应的权重
后面的count_vec.transform把文本按照前面的方法制作成相应的矩阵，但是没有进一部的操作，原因是留给后面的分类算法服务。
'''
#print x
#print y
#print x_train
print u'测试数据\n',x_test

########################################################################################
#######下面调用贝叶斯算法进行分类预测
########################################################################################
#调用MultinomialNB，有多种贝叶斯算法：
##from sklearn.naive_bayes import GaussianNB
##这里选取的贝叶斯的选择器是使用高斯分布的特征值
##from sklearn.naive_bayes import BernoulliNB
##这里选取的是贝叶斯的白努力分布（二值分布）
########################################################################################
clf=MultinomialNB()
clf.fit(x_train,y_train)
doc_class_predicted=clf.predict(x_test)

print count_vec.get_feature_names()#这是一个获取相应的特征值的
print u'训练集\n',x_train.toarray()
print u'预测的结果\n',doc_class_predicted
print u'实际值\n',y
print u'平均成功概率\n',np.mean(doc_class_predicted==y_test)

precision,recall,thresholds=precision_recall_curve(y_test,clf.predict(x_test))
answer=clf.predict_proba(x_test)[:,1]
'''
上面一句口令的意思是显示测试样本属于不同类别的概率分别是多少，其中[:,0]表示错误的概率，[:,1]表示正确的概率
'''
#print answer
report=answer >0.5
'''
显然report计算之后是布尔值，为什么选择0.5是因为相应的我们一般认为超过一半的可能，就认为发生的概率比较大
'''
#print report
print classification_report(y_test,report,target_names=['neg','pos'])
'''
举个栗子解释下：
测试结果：array([ 0.,  1.,  0.,  1.,  0.,  1.,  0.,  1.,  0.,  0.])
真实结果：array([ 0.,  1.,  0.,  1.,  0.,  1.,  0.,  1.,  0.,  1.])
其中0表示thin （瘦子）1表示fat（胖子）
分为thin的准确率为0.83。是因为分类器分出了6个thin，其中正确的有5个，因此分为thin的准确率为5/6=0.83。
分为thin的召回率为1.00。是因为数据集中共有5个thin，而分类器把他们都分对了（虽然把一个fat分成了thin！），召回率5/5=1。
分为fat的准确率为1.00。
分为fat的召回率为0.80。是因为数据集中共有5个fat，而分类器只分出了4个（把一个fat分成了thin！），召回率4/5=0.80。
很多时候，尤其是数据分类难度较大的情况，准确率与召回率往往是矛盾的。你可能需要根据你的需要找到最佳的一个平衡点。
比如本例中，你的目标是尽可能保证找出来的胖子是真胖子（准确率），还是保证尽可能找到更多的胖子（召回率）。
'''
