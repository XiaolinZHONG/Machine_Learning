#coding=utf-8

from sklearn.datasets import load_files
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


movie_reviews=load_files('d:\Pycharm\\tokens')
'''
支持从目录读取所有分类好的文本。不过目录必须按照一个文件夹一个标签名的规则放好。
比如本文使用的数据集共有2个标签，一个为“net”，一个为“pos”，每个目录下面有很多个文本文件。
'''
doc_terms_train, doc_terms_test, y_train, y_test\
    = train_test_split(movie_reviews.data, movie_reviews.target, test_size = 0.2)
'''
TF（词频）的计算很简单，就是针对一个文件t，某个单词Nt 出现在该文档中的频率。
比如文档“I love this movie”，单词“love”的TF为1/4。如果去掉停用词“I"和”it“，则为1/2。
IDF（逆向文件频率）的意义是，对于某个单词t，可以由总文件数目除以包含该词语之文件的数目，再将得到的商取对数得到。
IDF是为了凸显那种出现的少，但是占有强烈感情色彩的词语。
比如“movie”这样的词的IDF=ln(12/5)=0.88，远小于“love”的IDF=ln(12/1)=2.48。
'''
count_vec=TfidfVectorizer(binary=False,decode_error='ignore',stop_words='english')
'''
stop_words = 'english'，表示使用默认的英文停用词。可以使用count_vec.get_stop_words()查看TfidfVectorizer内置的所有停用词。
当然，在这里可以传递你自己的停用词list（比如这里的“movie”）
注意这些数据集可能存在非法字符问题。所以在构造count_vec时，传入了decode_error = 'ignore'，以忽略这些非法字符。
count_vec构造时默认传递了max_df=1，因此TF-IDF都做了规格化处理，以便将所有值约束在[0,1]之间。
'''
x_train=count_vec.fit_transform(doc_terms_train)
x_test=count_vec.transform(doc_terms_test)
x=count_vec.transform(movie_reviews.data)
y=movie_reviews.target

print doc_terms_train
print count_vec.get_feature_names()#这是一个获取相应的特征值的
print x_train.toarray()
print movie_reviews.target
'''
最后得到的结果是一个非常稀疏的矩阵
'''