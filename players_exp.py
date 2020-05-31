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
    num_points = (delta+1)**args.dim
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
    truth = [ ranking_count[i][1]/players for i in range(len(ranking_count)) ]
    result = [ result_ranking_count[i][1]/num_points for i in range(len(result_ranking_count)) ]
    ## 
    ## 相関係数
    cor = correlation(truth,result)
    return [cor,truth,result]

if __name__ == '__main__':
    # input value
    parser = argparse.ArgumentParser()
    parser.add_argument('-mp', '--min_np', type=int, help='実験するplayer数の最小値', default=100)
    parser.add_argument('-Mp', '--Max_np', type=int, help='実験するplayer数の最大値', default=10000)
    parser.add_argument('-mrp', '--min_rp', type=float, help='-rpの最小値', default=1e-9)
    parser.add_argument('-nrp', '--num_rp', type=int, help='-rpの実験個数', default=1)
    parser.add_argument('-mrb', '--min_rb', type=float, help='-rbの最小値', default=0.01)
    parser.add_argument('-nrb', '--num_rb', type=int, help='-rbの実験個数', default=1)
    parser.add_argument('-nb', type=int, help='ブランド数', default=5)
    parser.add_argument('-d', '--dim', type=int, help='次元', default=2)
    parser.add_argument('-p', '--path', type=str, help='今回の実験で使用するディレクトリ', default='./0422_players')
    parser.add_argument('-e', '--epoch', type=int, help='実験epoch数', default=200)
    parser.add_argument('-n', type=int, help='並列数', default=4)
    parser.add_argument('-del', '--delta', type=int, help='面積推定時の座標分割数', default=400)
    parser.add_argument('-s', '--plotshow', action='store_true', help='graph表示するか？')
    parser.add_argument('-ta', '--dimtest_actual', action='store_true', help='dim_exp.py実行かつ実データ？')
    parser.add_argument('-tt', '--dimtest_toy', action='store_true', help='dim_exp.py実行かつtoydata？')
    args = parser.parse_args()

    # def.
    path = args.path
    data_path = path+'/data'
    result_path = path+'/result/dim_'+str(args.dim)
    # [実験用] //////////////////////////////////////////
    if args.nb == 3:
        listp = [1713,3145,5377,7167,9055,10894,12384,14413,16103,18607] # 3ブランド選抜
    elif args.nb == 4:
        listp = [837,1673,2347,3235,4040,4837,5551,6385,7257,8102] # 4ブランド選抜
    elif args.nb == 5:
        listp = [504,1145,1631,2081,2589,3025,3489,3931,4389,5455] # 5ブランド選抜
    elif args.nb == 6:
        listp = [415,685,1003,1337,1599,1934,2254,2562,2883,3195] # 6ブランド選抜
    else:
        listp = [ args.min_np*(10**i) for i in range(int(math.log10(args.Max_np))-int(math.log10(args.min_np))+1) ]
    """ 0521更新
    if args.nb == 3:
        listp = [626,1083,1596,1713,2180,2726,2842,3145,4363,4511,4664,5055,5377,5500,5911,6687,6961,7167,7288,7654,7904,8029,8352,8553,9055,9860,9940,10749,10894,11314,11529,11907,12196,12384,13688,14109,14413,15122,16103,16978,18607] # 3ブランド選抜
    elif args.nb == 4:
        listp = [551,837,938,1295,1368,1673,2121,2210,2347,3110,3235,3347,3381,3718,3948,4040,4619,4837,4920,4996,5118,5300,5551,5758,5932,6106,6385,6418,6599,6701,6831,7125,7257,7366,7505,7781,8102] # 4ブランド選抜
    elif args.nb == 5:
        listp = [504,757,834,1145,1186,1402,1631,1790,1880,1918,2081,2139,2281,2341,2438,2589,2673,2700,2856,2994,3025,3202,3296,3376,3400,3489,3544,3672,3755,3837,3931,4196,4224,4389,5455] # 5ブランド選抜
    """
#    listp = [504,757,834,1145,1146,1186,1402,1631,1786,1790,1863,1880,1881,1918,2081,2139,2281,2341,2438,2589,2598,2611,2623,2649,2658,2673,2699,2700,2709,2856,2994,2996,3016,3025,3202,3244,3296,3328,3376,3400,3489,3544,3672,3755,3790,3837,3886,3931,4196,4224,4389,5455] # 5ブランド
#    listp = [551,837,938,1295,1296,1368,1673,2121,2126,2210,2234,2235,2347,3110,3235,3336,3347,3351,3381,3718,3948,4028,4040,4619,4837,4920,4996,5118,5300,5351,5551,5555,5758,5932,6106,6385,6418,6599,6610,6694,6701,6815,6823,6831,7125,7186,7257,7263,7366,7505,7781,8102] # 4ブランド
#    listp = [626,1004,1083,1596,1597,1713,2180,2726,2732,2807,2839,2842,3145,4363,4432,4511,4532,4614,4664,5055,5377,5500,5911,6687,6961,7167,7288,7654,7904,8029,8352,8553,9055,9860,9868,9940,10749,10894,11314,11336,11529,11907,12123,12196,12384,13688,14109,14413,15122,16103,16978,18607] # 3ブランド
    # //////////////////////////////////////////////////
    log_listp = [ math.log10(i) for i in listp ]
    listrp = [ float(args.min_rp * (10 ** i)) for i in range(args.num_rp) ] # listrp = [mrp,mrp*10,...,mrp*10^(nrp-1)]
    listrb = [ float(args.min_rb * (10 ** i)) for i in range(args.num_rb) ]
    savelist = []

    # graph set
    fig, ax = plt.subplots(1, args.num_rp, figsize=(7,7), squeeze=False)
    
    # experiment
    for rp in listrp:
        # graph set
        ax[0,int(listrp.index(rp))].set_title('dim:'+str(args.dim)+'/ rp:'+str(rp))
        ax[0,int(listrp.index(rp))].set_xlabel('players')
        ax[0,int(listrp.index(rp))].set_ylabel('correlation coefficient')
        ax[0,int(listrp.index(rp))].set_ylim(-1.0,1.0)
        for rb in listrb:
            cor_list = []
            act_cor_list = []
            for np in listp:
                # toy data
                if not args.dimtest_actual:
                    ## arrangement.py の実行 or するー
                    data_dir = data_path+'/'+str(args.nb)+'_'+str(np)
                    if not os.path.exists(data_dir):
                        ### file名(日付_時刻)の取得
                        dt = datetime.datetime.now()
                        filename = str('{0:02d}'.format(dt.month)+'{0:02d}'.format(dt.day)+'_'+'{0:02d}'.format(dt.hour)+'{0:02d}'.format(dt.minute))
                        if dt.minute == 59:
                            filename2 = str('{0:02d}'.format(dt.month)+'{0:02d}'.format(dt.day)+'_'+'{0:02d}'.format(dt.hour+1)+'00')
                        else:
                            filename2 = str('{0:02d}'.format(dt.month)+'{0:02d}'.format(dt.day)+'_'+'{0:02d}'.format(dt.hour)+'{0:02d}'.format(dt.minute+1))
                        ### arrangement.py -> brands.csv & ranking.csv & full_ranking.csv の作成
                        os.system(
                            'python arrangement.py -nb '+str(args.nb)+' -np '+str(np)+' -g --dim '+str(args.dim)+' -o '+data_path
                        )
                        ### dir name の変更 date -> "nb_np"
                        if os.path.exists(data_path+'/'+filename):
                            os.rename(data_path+'/'+filename,data_dir)
                        elif os.path.exists(data_path+filename2):
                            os.rename(data_path+'/'+filename2,data_dir)
                    ## newrankLearn.py の実行 or するー
                    result_dir = result_path+'/epoch_'+str(args.epoch)+'/'+str(args.nb)+'_'+str(np)
                    if not os.path.exists(result_dir+'/'+str(rp)+'_'+str(rb)):
                        ## file名(日付_時刻)の取得
                        dt = datetime.datetime.now()
                        filename = str('{0:02d}'.format(dt.month)+'{0:02d}'.format(dt.day)+'_'+'{0:02d}'.format(dt.hour)+'{0:02d}'.format(dt.minute))
                        if dt.minute == 59:
                            filename2 = str('{0:02d}'.format(dt.month)+'{0:02d}'.format(dt.day)+'_'+'{0:02d}'.format(dt.hour+1)+'00')
                        else:
                            filename2 = str('{0:02d}'.format(dt.month)+'{0:02d}'.format(dt.day)+'_'+'{0:02d}'.format(dt.hour)+'{0:02d}'.format(dt.minute+1))
                        ## newrankLearn.py
                        os.system(
                            'mpiexec -n '+str(args.n)+' python newrankLearn.py '+data_dir+'/ranking.csv -e '+str(int(args.epoch/args.n))+' -rp '+str(rp)+' -rb '+str(rb)+' -d '+str(args.dim)+' -o '+str(result_dir)+' --mpi'
                        )
                        ## dir name の変更 date -> "nb_np"
                        if os.path.exists(result_dir+'/'+filename):
                            os.rename(result_dir+'/'+filename,result_dir+'/'+str(rp)+'_'+str(rb))
                        elif os.path.exists(result_dir+'/'+filename2):
                            os.rename(result_dir+'/'+filename2,result_dir+'/'+str(rp)+'_'+str(rb))
                    ## 検証
                    if os.path.exists(result_dir+'/'+str(rp)+'_'+str(rb)+'/cor.csv') and os.path.exists(result_dir+'/'+str(rp)+'_'+str(rb)+'/truth_area.csv') and os.path.exists(result_dir+'/'+str(rp)+'_'+str(rb)+'/result_area.csv'):
                        ### csv open
                        list11 = csv_open(result_dir+'/'+str(rp)+'_'+str(rb)+'/cor.csv')
                        ### csv -> float value
                        cor = [ list(map(float,list11[i])) for i in range(len(list11)) ][0][0]
                    else:
                        ### csv open
                        list1 = csv_open(data_dir+'/full_ranking.csv')
                        list2 = csv_open(result_dir+'/'+str(rp)+'_'+str(rb)+'/out_brands0000.csv')
                        ### csv -> list
                        fullranking = [ list(map(int,list1[i])) for i in range(len(list1)) ]
                        result_brands_points = [ list(map(float,list2[i])) for i in range(len(list2)) ]
                        ### 相関係数の導出
                        evaluate_list = evaluate(fullranking,result_brands_points,args)
                        cor = evaluate_list[0]
                        truth_area = [evaluate_list[1]]
                        result_area = [evaluate_list[2]]
                        ### Save cor.csv
                        cor_csv_savelist = [[cor]]
                        csv_save(cor_csv_savelist,result_dir+'/'+str(rp)+'_'+str(rb)+'/cor.csv')
                        csv_save(truth_area,result_dir+'/'+str(rp)+'_'+str(rb)+'/truth_area.csv')
                        csv_save(result_area,result_dir+'/'+str(rp)+'_'+str(rb)+'/result_area.csv')
                    cor_list.append(cor)
                    print('dim:',str(args.dim),'/ rp:',str(rp),'/ rb:',str(rb),'/ ',str(np)+'players:',cor)
                if not args.dimtest_toy:
                    ## 実データからの選抜
                    actual_data_dir = path+'/actual_data/'+str(args.nb)+'_'+str(np)
                    ## newrankLearn.py の実行 or するー
                    actual_result_dir = path+'/actual_result/dim_'+str(args.dim)+'/epoch_'+str(args.epoch)+'/'+str(args.nb)+'_'+str(np)
                    if not os.path.exists(actual_result_dir+'/'+str(rp)+'_'+str(rb)):
                        ## file名(日付_時刻)の取得
                        act_dt = datetime.datetime.now()
                        act_filename = str('{0:02d}'.format(act_dt.month)+'{0:02d}'.format(act_dt.day)+'_'+'{0:02d}'.format(act_dt.hour)+'{0:02d}'.format(act_dt.minute))
                        if act_dt.minute == 59:
                            act_filename2 = str('{0:02d}'.format(act_dt.month)+'{0:02d}'.format(act_dt.day)+'_'+'{0:02d}'.format(act_dt.hour+1)+'00')
                        else:
                            act_filename2 = str('{0:02d}'.format(act_dt.month)+'{0:02d}'.format(act_dt.day)+'_'+'{0:02d}'.format(act_dt.hour)+'{0:02d}'.format(act_dt.minute+1))
                        ## newrankLearn.py
                        os.system(
                            'mpiexec -n '+str(args.n)+' python newrankLearn.py '+actual_data_dir+'/ranking.csv -e '+str(int(args.epoch/args.n))+' -rp '+str(rp)+' -rb '+str(rb)+' -d '+str(args.dim)+' -o '+str(actual_result_dir)+' --mpi'
                        )
                        ## dir name の変更 date -> "nb_np"
                        if os.path.exists(actual_result_dir+'/'+act_filename):
                            os.rename(actual_result_dir+'/'+act_filename,actual_result_dir+'/'+str(rp)+'_'+str(rb))
                        elif os.path.exists(actual_result_dir+'/'+act_filename2):
                            os.rename(actual_result_dir+'/'+act_filename2,actual_result_dir+'/'+str(rp)+'_'+str(rb))
                    ## 検証 or すでにしたやつを読み込み
                    if os.path.exists(actual_result_dir+'/'+str(rp)+'_'+str(rb)+'/cor.csv') and os.path.exists(actual_result_dir+'/'+str(rp)+'_'+str(rb)+'/truth_area.csv') and os.path.exists(actual_result_dir+'/'+str(rp)+'_'+str(rb)+'/result_area.csv'):
                        ### csv open
                        list21 = csv_open(actual_result_dir+'/'+str(rp)+'_'+str(rb)+'/cor.csv')
                        ### csv -> float value
                        act_cor = [ list(map(float,list21[i])) for i in range(len(list21)) ][0][0]
                    else:
                        ### csv open
                        list3 = csv_open(actual_data_dir+'/full_ranking.csv')
                        list4 = csv_open(actual_result_dir+'/'+str(rp)+'_'+str(rb)+'/out_brands0000.csv')
                        ### csv -> list
                        actual_fullranking = [ list(map(int,list3[i])) for i in range(len(list3)) ]
                        actual_result_brands_points = [ list(map(float,list4[i])) for i in range(len(list4)) ]
                        ### 相関係数の導出
                        act_evaluate_list = evaluate(actual_fullranking,actual_result_brands_points,args)
                        act_cor = act_evaluate_list[0]
                        act_truth_area = [act_evaluate_list[1]]
                        act_result_area = [act_evaluate_list[2]]
                        ### Save cor.csv
                        act_cor_csv_savelist = [[act_cor]]
                        csv_save(act_cor_csv_savelist,actual_result_dir+'/'+str(rp)+'_'+str(rb)+'/cor.csv')
                        csv_save(act_truth_area,actual_result_dir+'/'+str(rp)+'_'+str(rb)+'/truth_area.csv')
                        csv_save(act_result_area,actual_result_dir+'/'+str(rp)+'_'+str(rb)+'/result_area.csv')
                    act_cor_list.append(act_cor)
                    print('dim:',str(args.dim),'/ rp:',str(rp),'/ rb:',str(rb),'/ ',str(np)+'players (actual) :',act_cor)
            # graph plot
            if not args.dimtest_actual:
                ax[0,int(listrp.index(rp))].plot(listp,cor_list,label='rb:'+str(rb),marker='.',linestyle="dashed")
            elif not args.dimtest_toy:
                ax[0,int(listrp.index(rp))].plot(listp,act_cor_list,label='act_rb:'+str(rb),marker='.')
            # savelist
            if not args.dimtest_actual:
                cor_list = []
            elif not args.dimtest_toy:
                act_cor_list = []
            savelist.append([args.dim,rp,rb,args.epoch,log_listp,cor_list,act_cor_list])
    ####### あとで実装
    # graph
    fig.legend(loc='lower right')
    fig.tight_layout()
    fig.align_labels()
    if args.plotshow:
        plt.show()
    # Save
    summary_data = csv_open(path+'/summary.csv')
    summary_data.append(savelist)
    csv_save(summary_data,path+'/summary.csv')


