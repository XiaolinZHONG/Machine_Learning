#coding=utf-8
import numpy as np
import scipy as sp

xx,yy=np.meshgrid(np.arange(1,5,0.1),np.arange(20,90,1))
print xx.shape,'\n',xx
print yy.shape,'\n',yy
print type(xx)

probality=np.hstack((xx.ravel(),yy.ravel()))
print xx.ravel().shape,xx.ravel(),'\n',xx.ravel().shape,yy.ravel()
print probality
probality=np.c_[xx.ravel(),yy.ravel()]
print probality
print type(probality)

print '#'*25,'\n'

a=np.arange(15).reshape(3,5)
b=a+np.random.uniform(0,1)
print a,b
print '*'*25,'\n'
print a.ravel(),'\n',b.ravel(),'\n'
print '*'*25,'\n'
print np.hstack((a,b))
print '*'*25,'\n'
print np.c_[a,b]
print '*'*25,'\n'
print np.hstack((a.ravel(),b.ravel()))
print '*'*25,'\n'
print np.c_[a.ravel(),b.ravel()]