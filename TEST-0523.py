#coding=utf-8
import os
import jieba
import jieba.posseg as pseg
import sys
import string
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
reload(sys)
sys.setdefaultencoding('utf8')
#获取文件列表（该目录下放着100份文档）
def getFilelist(path) :
    filelist = []
    for root,dirs,item in os.walk(path):
        for f in item :
            filelist.append(f)
        return filelist
    print filelist
#对文档进行分词处理
def fenci(ff,path) :
    #保存分词结果的目录
    sFilePath = 'D:\\anaconda project\TEST1'
    stopkey=[line.strip().decode('utf-8') for line in open('stopwords.txt').readlines()]
    if not os.path.exists(sFilePath) :
        os.mkdir(sFilePath)
    #读取文档
    filename = ff
    f = open(path+filename,'r')
    file_list = f.read()
    f.close()

    #对文档进行分词处理，采用默认模式
    seg_list = jieba.cut(file_list,cut_all=True)



    #对空格，换行符进行处理
    result = []
    for seg in seg_list :
        seg = ''.join(list(set(seg)-set(stopkey)))
        if (seg != '' and seg != "\n" and seg != "\n\n") :
            result.append(seg)

    #将分词后的结果用空格隔开，保存至本地。比如"我来到北京清华大学"，分词结果写入为："我 来到 北京 清华大学"
    f = open(sFilePath+"/"+filename+"-seg.txt","w+")
    f.write(' '.join(result))
    f.close()

#读取100份已分词好的文档，进行TF-IDF计算
def Tfidf(filelist) :
    path = 'D:\\anaconda project\TEST1\\'
    corpus = []  #存取100份文档的分词结果
    for ff in filelist :
        fname = path + ff+"-seg.txt"
        f = open(fname,'r+')
        content = f.read()
        f.close()
        corpus.append(content)

    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))

    word = vectorizer.get_feature_names() #所有文本的关键字
    weight = tfidf.toarray()              #对应的tfidf矩阵

    sFilePath = 'D:\\anaconda project\TEST2\\'
    if not os.path.exists(sFilePath) :
        os.mkdir(sFilePath)

    # 这里将每份文档词语的TF-IDF写入tfidffile文件夹中保存
    for i in range(len(weight)) :
        print u"--Writing all the tf-idf in the",i,u" file into ",sFilePath+'\\'+string.zfill(i,5)+'.txt',"--"
        f = open(sFilePath+'/'+string.zfill(i,5)+'.txt','w+')
        for j in range(len(word)) :
            f.write(word[j]+"    "+str(weight[i][j])+"\n")
        f.close()

if __name__ == "__main__":
    path='D:\\anaconda project\TEST\\'
    allfile= getFilelist(path)
    for ff in allfile:
        print "Using jieba on "+ff
        fenci(ff,path)
    Tfidf(allfile)