#coding=utf-8
import numpy as np
import scipy as sp
from sklearn import svm
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
xx,yy=np.meshgrid(np.arange(1,5,0.1),np.arange(20,90,1))
print xx
print yy