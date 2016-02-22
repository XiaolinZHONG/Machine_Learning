# -*- coding: utf-8 -*-  
from matplotlib import pyplot  
import scipy as sp  
import numpy as np  
from matplotlib import pylab  
from sklearn.datasets import load_files  
from sklearn.cross_validation import train_test_split  
from sklearn.metrics import precision_recall_curve, roc_curve, auc  
from sklearn.metrics import classification_report
import time
from scipy import sparse

start_time=time.time()
