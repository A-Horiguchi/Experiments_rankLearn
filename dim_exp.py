# coding : utf-8

import csv
import pandas as pd
import numpy as np
import math
import random
import matplotlib
import matplotlib.pyplot as plt
import itertools as it
import os
import datetime
import re
import copy
import sys
from tqdm import tqdm
from operator import itemgetter
import pprint
import itertools
import argparse

def csv_open(loadpath):
    with open(loadpath) as file:
        reader = csv.reader(file)
        result = [row for row in reader]
    return result

def csv_save(savelist,savepath):
    with open(savepath, 'w') as file:
        writer = csv.writer(file)
        writer.writerows(savelist)

def distance(list1,list2):
    d = 0
    for i in range(len(list1)):
        d += (list1[i]-list2[i]) ** 2
    d = math.sqrt(d)
    return d

def correlation(list1,list2):
    s1 = pd.Series(list1)
    s2 = pd.Series(list2)
    res = s1.corr(s2)
    return res

def evaluate(fullranking,result_brands_points,args):
    # rankingの総順列list brands_permutation の作成
    list_1 = [str(i) for i in range(args.nb)]
    list_2 = list(itertools.permutations(list_1))
    brands_permutation = [ ''.join(list_2[i]) for i in range(len(list_2)) ]
    permu_terms = len(list_2)
    # ranking_count の作成
    ranking_count = [ [brands_permutation[i],0] for i in range(len(brands_permutation)) ]
    ## fullranking のカウント
    players = len(fullranking)
    for i in range(players):
        list_i = list(map(str,fullranking[i]))
        element = ''.join(list_i)
        if element in brands_permutation:
            ranking_count[int(brands_permutation.index(element))][1] += 1
    # result_ranking_count の作成
    result_ranking_count = [ [brands_permutation[i],0] for i in range(len(brands_permutation)) ]
    ## 各点(points)の距離順列
    delta = args.delta
    coordinates = [-1+2*(i/delta) for i in range(delta+1)] #[-1,1]をdelta分割
    num_points = (delta+1)**2
    if args.dim == 2:
        for x1,x2 in tqdm(itertools.product(coordinates,coordinates)):
            point = [x1,x2]
            distance_list = []
            for l in range(len(result_brands_points)):
                distance_list.append([l,distance(point,result_brands_points[l])])
            distance_list.sort(key=itemgetter(1))
            distance_rank = [ str(distance_list[i][0]) for i in range(len(distance_list)) ]
            result_element = ''.join(distance_rank)
            if result_element in brands_permutation:
                result_ranking_count[int(brands_permutation.index(result_element))][1] += 1
    elif args.dim == 3:
        for x1,x2,x3 in tqdm(itertools.product(coordinates,coordinates,coordinates)):
            point = [x1,x2,x3]
            distance_list = []
            for l in range(len(result_brands_points)):
                distance_list.append([l,distance(point,result_brands_points[l])])
            distance_list.sort(key=itemgetter(1))
            distance_rank = [ str(distance_list[i][0]) for i in range(len(distance_list)) ]
            result_element = ''.join(distance_rank)
            if result_element in brands_permutation:
                result_ranking_count[int(brands_permutation.index(result_element))][1] += 1
    elif args.dim == 4:
        for x1,x2,x3,x4 in tqdm(itertools.product(coordinates,coordinates,coordinates,coordinates)):
            point = [x1,x2,x3,x4]
            distance_list = []
            for l in range(len(result_brands_points)):
                distance_list.append([l,distance(point,result_brands_points[l])])
            distance_list.sort(key=itemgetter(1))
            distance_rank = [ str(distance_list[i][0]) for i in range(len(distance_list)) ]
            result_element = ''.join(distance_rank)
            if result_element in brands_permutation:
                result_ranking_count[int(brands_permutation.index(result_element))][1] += 1
    elif args.dim == 5:
        for x1,x2,x3,x4,x5 in tqdm(itertools.product(coordinates,coordinates,coordinates,coordinates,coordinates)):
            point = [x1,x2,x3,x4,x5]
            distance_list = []
            for l in range(len(result_brands_points)):
                distance_list.append([l,distance(point,result_brands_points[l])])
            distance_list.sort(key=itemgetter(1))
            distance_rank = [ str(distance_list[i][0]) for i in range(len(distance_list)) ]
            result_element = ''.join(distance_rank)
            if result_element in brands_permutation:
                result_ranking_count[int(brands_permutation.index(result_element))][1] += 1
    elif args.dim == 6:
        for x1,x2,x3,x4,x5,x6 in tqdm(itertools.product(coordinates,coordinates,coordinates,coordinates,coordinates,coordinates)):
            point = [x1,x2,x3,x4,x5,x6]
            distance_list = []
            for l in range(len(result_brands_points)):
                distance_list.append([l,distance(point,result_brands_points[l])])
            distance_list.sort(key=itemgetter(1))
            distance_rank = [ str(distance_list[i][0]) for i in range(len(distance_list)) ]
            result_element = ''.join(distance_rank)
            if result_element in brands_permutation:
                result_ranking_count[int(brands_permutation.index(result_element))][1] += 1
    elif args.dim == 7:
        for x1,x2,x3,x4,x5,x6,x7 in tqdm(itertools.product(coordinates,coordinates,coordinates,coordinates,coordinates,coordinates,coordinates)):
            point = [x1,x2,x3,x4,x5,x6,x7]
            distance_list = []
            for l in range(len(result_brands_points)):
                distance_list.append([l,distance(point,result_brands_points[l])])
            distance_list.sort(key=itemgetter(1))
            distance_rank = [ str(distance_list[i][0]) for i in range(len(distance_list)) ]
            result_element = ''.join(distance_rank)
            if result_element in brands_permutation:
                result_ranking_count[int(brands_permutation.index(result_element))][1] += 1
    else:
        print("This program is not compatible with dimention",str(args.dim))
    # 評価
    ## 比較listの作成
    truth = [ ranking_count[i][1]/num_points for i in range(len(ranking_count)) ]
    result = [ result_ranking_count[i][1]/players for i in range(len(result_ranking_count)) ]
    ## 相関係数
    cor = correlation(truth,result)
    return [cor,truth,result]

if __name__ == '__main__':
    # input value
    parser = argparse.ArgumentParser()
    parser.add_argument('-rp', type=float, help='-rp値', default=1e-09)
    parser.add_argument('-rb', type=float, help='-rb値', default=0.01)
    parser.add_argument('-nb', type=int, help='ブランド数', default=5)
    parser.add_argument('-md', '--min_dim', type=int, help='最小次元', default=2)
    parser.add_argument('-Md', '--Max_dim', type=int, help='最大次元', default=7)
    parser.add_argument('-dp', '--datapath', type=str, help='dataが格納されているdirectory', default='./0422_players')
#    parser.add_argument('-sp', '--savepath', type=str, help='結果を保存するdirectory', default='./0520_dim')
    parser.add_argument('-e', '--epoch', type=int, help='実験epoch数', default=200)
    parser.add_argument('-n', type=int, help='並列数', default=4)
#    parser.add_argument('-del', '--delta', type=int, help='面積推定時の座標分割数', default=200)
    parser.add_argument('-t', '--toydata', action='store_true', help='toydataでの実験？')
    args = parser.parse_args()

    # def.
    ## dim_list & delta_list
    dim_list = [i for i in range(args.min_dim,args.Max_dim+1)]
    delta_list = [10000,465,100,40,22,14,10]
    ## players
    if args.nb == 3:
        player_list = [1713,3145,5377,7167,9055,10894,12384,14413,16103,18607] # 3ブランド選抜
    elif args.nb == 4:
        player_list = [837,1673,2347,3235,4040,4837,5551,6385,7257,8102] # 4ブランド選抜
    elif args.nb == 5:
        player_list = [504,1145,1631,2081,2589,3025,3489,3931,4389,5455] # 5ブランド選抜
    elif args.nb == 6:
        player_list = [415,685,1003,1337,1599,1934,2254,2562,2883,3195] # 6ブランド選抜
    else:
        print("actual_exp.pyを実行してください．実データの選抜ができていません．")
        sys.exit()

    # graph set
    fig, ax = plt.subplots(1,1, figsize=(7,7), squeeze=False)
    ax[0,0].set_title('correlation coefficient - dimention @ '+str(args.nb)+'brands')
    ax[0,0].set_xlabel('number of players')
    ax[0,0].set_ylabel('correlation coefficient')
    ax[0,0].set_ylim(-1.0,1.0)

    # experiment
    for dim in dim_list:
        # delta
        if dim <= 8:
            delta = delta_list[int(dim-2)]
        else:
            delta = delta_list[-1]
        # main
        plotlist = []
        for player in player_list:
            if not args.toydata:
                ## rankLearn.pyを実行後の結果データが存在しない -> 実行
                if not os.path.exists(args.datapath+'/actual_result/dim_'+str(dim)+'/epoch_'+str(args.epoch)+'/'+str(args.nb)+'_'+str(player)+'/'+str(args.rp)+'_'+str(args.rb)+'/cor.csv'):
                    os.system('python players_exp.py -p '+str(args.datapath)+' -nb '+str(args.nb)+' -d '+str(dim)+' -del '+str(delta)+' -ta')
                ## cor.csvの読み込み
                list1 = csv_open(args.datapath+'/actual_result/dim_'+str(dim)+'/epoch_'+str(args.epoch)+'/'+str(args.nb)+'_'+str(player)+'/'+str(args.rp)+'_'+str(args.rb)+'/cor.csv')
                cor = [ list(map(float,list1[i])) for i in range(len(list1)) ][0][0]
                plotlist.append(cor)
            else:
                ## rankLearn.pyを実行後の結果データが存在しない -> 実行
                if not os.path.exists(args.datapath+'/result/dim_'+str(dim)+'/epoch_'+str(args.epoch)+'/'+str(args.nb)+'_'+str(player)+'/'+str(args.rp)+'_'+str(args.rb)+'/cor.csv'):
                    os.system('python players_exp.py -p '+str(args.datapath)+' -nb '+str(args.nb)+' -d '+str(dim)+' -del '+str(delta)+' -tt')
                ## cor.csvの読み込み
                list1 = csv_open(args.datapath+'/result/dim_'+str(dim)+'/epoch_'+str(args.epoch)+'/'+str(args.nb)+'_'+str(player)+'/'+str(args.rp)+'_'+str(args.rb)+'/cor.csv')
                cor = [ list(map(float,list1[i])) for i in range(len(list1)) ][0][0]
                plotlist.append(cor)
        ax[0,0].plot(player_list,plotlist,label='dim:'+str(dim),marker='.')

    # graph
    fig.legend(loc='lower right')
    fig.tight_layout()
    fig.align_labels()
    plt.show()


