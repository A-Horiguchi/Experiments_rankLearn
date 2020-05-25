# coding: utf-8

import csv
import pandas as pd
import numpy as np
import math
import random
import matplotlib
import matplotlib.pyplot as plt
import itertools as it
import os
import glob
import datetime
import re
import copy
import sys
from tqdm import tqdm
from operator import itemgetter
import pprint
import itertools
import argparse
import time

#listp = [1713,3145,5377,7167,9055,10894,12384,14413,16103,18607] # 3ブランド選抜
listp = [837,1673,2347,3235,4040,4837,5551,6385,7257,8102] # 4ブランド選抜
#listp = [504,1145,1631,2081,2589,3025,3489,3931,4389,5455] # 5ブランド選抜

fig, ax = plt.subplots(2, 5, figsize=(12,7), squeeze=False)

for player in listp:
    with open('./0422_players/actual_result/dim_5/epoch_200/4_'+str(player)+'/1e-09_0.01/result_area.csv') as file:
        reader = csv.reader(file)
        result = [row for row in reader]

    data1 = [ list(map(float,result[i])) for i in range(len(result)) ]

    with open('./0422_players/actual_result/dim_5/epoch_200/4_'+str(player)+'/1e-09_0.01/truth_area.csv') as file:
        reader = csv.reader(file)
        truth = [row for row in reader]

    data2 = [ list(map(float,truth[i])) for i in range(len(truth)) ]

    npx = np.array(data1)
    npy = np.array(data2)
    
    test = np.nanmax(npx)
    
    Max = max(np.nanmax(npx),np.nanmax(npy))

    x = np.linspace(-0.005,Max*1.1,100)
    y = x

    if int(listp.index(player)) <= 4:
        ax[0,int(listp.index(player))].set_title(str(player)+' players')
        ax[0,int(listp.index(player))].set_xlabel('result')
        ax[0,int(listp.index(player))].set_ylabel('truth')
        ax[0,int(listp.index(player))].scatter(data1,data2,marker='.')
        ax[0,int(listp.index(player))].plot(x,y,linewidth=0.5)
        ax[0,int(listp.index(player))].set_xlim(-0.005,Max*1.1)
        ax[0,int(listp.index(player))].set_ylim(-0.005,Max*1.1)
    else:
        ax[1,int(listp.index(player)-5)].set_title(str(player)+' players')
        ax[1,int(listp.index(player)-5)].set_xlabel('result')
        ax[1,int(listp.index(player)-5)].set_ylabel('truth')
        ax[1,int(listp.index(player)-5)].scatter(data1,data2,marker='.')
        ax[1,int(listp.index(player)-5)].plot(x,y,linewidth=0.5)
        ax[1,int(listp.index(player)-5)].set_xlim(-0.005,Max*1.1)
        ax[1,int(listp.index(player)-5)].set_ylim(-0.005,Max*1.1)

fig.tight_layout()
plt.show()



