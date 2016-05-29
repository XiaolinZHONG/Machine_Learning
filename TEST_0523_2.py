#coding=utf-8
import scipy as sp
import numpy as np
import sys

from sklearn.datasets import load_files
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import   TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report

reload(sys)
sys.setdefaultencoding('utf-8')
commands=load_files('D:\\anaconda project\pn\\')
#print commands
doc_terms_train, doc_terms_test, y_train, y_test\
    = train_test_split(commands.data, commands.target, test_size = 0.1)
stopkey=[line.strip().decode('utf-8') for line in open('stopwords.txt').readlines()]



count_vec = TfidfVectorizer(binary = False, decode_error = 'ignore',stop_words=stopkey)
x_train= count_vec.fit_transform(doc_terms_train)
x_test= count_vec.transform(doc_terms_test)
x= count_vec.transform(commands.data)
y= commands.target
#print(doc_terms_train)
a=count_vec.get_feature_names()
print count_vec.get_feature_names()
print a[0],type(a[1])
#print(x_train.toarray())
#print(commands.target)

clf = MultinomialNB().fit(x_train, y_train)
print x_train,x_test
doc_class_predicted = clf.predict(x_test)

#print(doc_class_predicted)
#print(y)
#print(np.mean(doc_class_predicted == y_test))

#准确率与召回率
precision, recall, thresholds = precision_recall_curve(y_test, clf.predict(x_test))
answer = clf.predict_proba(x_test)[:,1]
report = answer > 0.5
print(classification_report(y_test, report, target_names = ['neg', 'pos']))