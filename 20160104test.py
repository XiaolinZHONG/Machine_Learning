#coding=utf-8
from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import BeautifulSoup
import requests
from pattern import web
import scipy.stats as stats
import statsmodels.api as sm
from scipy.stats import binom

import re
from StringIO import StringIO
from zipfile import ZipFile
from pandas import read_csv

#nice defaults for matplotlib
from matplotlib import rcParams
dark2_colors = [(0.10588235294117647, 0.6196078431372549, 0.4666666666666667),
                (0.8509803921568627, 0.37254901960784315, 0.00784313725490196),
                (0.4588235294117647, 0.4392156862745098, 0.7019607843137254),
                (0.9058823529411765, 0.1607843137254902, 0.5411764705882353),
                (0.4, 0.6509803921568628, 0.11764705882352941),
                (0.9019607843137255, 0.6705882352941176, 0.00784313725490196),
                (0.6509803921568628, 0.4627450980392157, 0.11372549019607843),
                (0.4, 0.4, 0.4)]

rcParams['figure.figsize'] = (10, 6)
rcParams['figure.dpi'] = 150
rcParams['axes.color_cycle'] = dark2_colors
rcParams['lines.linewidth'] = 2
rcParams['axes.grid'] = True
rcParams['axes.facecolor'] = '#eeeeee'
rcParams['font.size'] = 14
rcParams['patch.edgecolor'] = 'none'

zip_folder = requests.get('http://seanlahman.com/files/database/lahman-csv_2014-02-14.zip').content
zip_files = StringIO()
zip_files.write(zip_folder)
csv_files = ZipFile(zip_files)

teams = csv_files.open('Teams.csv')
teams = read_csv(teams)
#print teams
players = csv_files.open('Batting.csv')
players = read_csv(players)
#print players
salaries = csv_files.open('Salaries.csv')
salaries = read_csv(salaries)


dat = teams[(teams['G'] == 162)&(teams['yearID']<2002) ]

print dat[["teamID","yearID","H","2B","3B","HR","BB"]].head()
players=players[players['yearID']>=1947]
def f(series):
    return series.index[np.where(series==min(series))][0]
df=players[players['AB']>502]
df=pd.df
grouped=df.goupby('playerID',as_index=False)
print grouped
rookie_index=grouped['yearID'].aggregate()
