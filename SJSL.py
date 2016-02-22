#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from matplotlib.font_manager import FontProperties
font=FontProperties(fname=r'c:\windows\fonts\simsun.ttc',size=12)
x=np.random.uniform(1,100,1000)#在1-100之间随机生成1000个数
y=np.log(x)+np.random.uniform(-0.5,.5,1000)
x1=np.arange(1,100,1)
plt.plot(x,y,'k.')
plt.scatter(x,y,s=3)#这一句和前面的一句的结果相同
plt.plot(x1,np.log(x1),'b')
plt.xlabel(u'这个是x轴',fontproperties=font)
plt.ylabel('this is y')
plt.xlim(-10,120)
plt.ylim(0,6)#可以缺省部分
plt.title('this is test for xl')
plt.grid(True)
plt.legend()
plt.show()
