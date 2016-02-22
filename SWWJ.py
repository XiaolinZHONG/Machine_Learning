#coding=utf-8
import re
import os
import xlwt

f=open('C:\Users\xzhon009\Desktop\回收问卷\\01 一年级-1班.xlsx')

remove1=re.compile('非常喜欢')
remove2=re.compile('较喜欢')
remove3=re.compile('不喜欢')

remove4=re.compile('非常满意')
remove5=re.compile('较满意')
remove6=re.compile('不满意')
def replacetool(x):
    x=re.sub(remove1,'3',x)
    x=re.sub(remove2,'2',x)
    x=re.sub(remove3,'1',x)
    x=re.sub(remove4,'3',x)
    x=re.sub(remove5,'2',x)
    x=re.sub(remove6,'1',x)
    return x.strip()

for i in

f.close()