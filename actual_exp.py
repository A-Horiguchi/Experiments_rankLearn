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

def csv_open(csvfile):
    with open(csvfile) as file:
        reader = csv.reader(file)
        result = [row for row in reader]
    return result

def csv_save(savelist,csvfile):
    with open(csvfile, 'w') as file:
        writer = csv.writer(file)
        writer.writerows(savelist)

def str_to_list(original_str,list_type):
    str1 = original_str.lstrip('[')
    str2 = str1.rstrip(']')
    str_list1 = str2.split(',')
    if str(list_type) == 'int':
        result = [int(str_list1[i]) for i in range(len(str_list1))]
    elif str(list_type) == 'float':
        result = [float(str_list1[i]) for i in range(len(str_list1))]
    elif str(list_type) == 'str':
        result = [str(str_list1[i]) for i in range(len(str_list1))]
    else:
        print("What's list_type?")
    return result

def arrangement(list_in_list): # return -> [list_in_list,num_players,num_brands]
    list_in_list.pop(0)
    list_1 = list_in_list
    list_2 = sorted(list_1,key=itemgetter(0))
    # numbering
    ## const.
    organize = []
    players_name = []
    brands_name = []
    ## 作成
    for i in range(len(list_2)):
        ### cnt
        ith_cnt = int(list_2[i][2])
        ### id & brands
        if i == 0:
            players_name.append([0,list_2[i][0]]) # players
            brands_name.append(list_2[i][1]) # brands
            organize.append([0,brands_name.index(list_2[i][1]),ith_cnt])
        else:
            #### id
            if list_2[i][0] == list_2[i-1][0]:
                ith_id = organize[i-1][0]
            else:
                ith_id = organize[i-1][0] + 1
                players_name.append([ith_id,list_2[i][0]])
            #### brands
            if not list_2[i][1] in brands_name:
                brands_name.append(list_2[i][1])
            ith_brand = brands_name.index(list_2[i][1])
            organize.append([ith_id,ith_brand,ith_cnt])
    list_3 = organize
    # making list ([id,b1,b2,...,b25])
    list_4 = [ [i] for i in range(int(list_3[-1][0])+1)]
    for i in range(len(list_4)):
        for j in range(len(brands_name)):
            list_4[i].append(0)
    id_data_list = [list_3[i][0] for i in range(len(list_3))]
    for i in range(len(id_data_list)):
        list_4[int(list_3[i][0])][int(list_3[i][1])+1] += int(list_3[i][2])
    return [list_4,len(players_name),len(brands_name)]

def condition(list_in_list,a,b,c,d,num_brands,condition_brands): # return -> [list_in_list,選抜brands' ids,全brandの購入数list,選抜brandの購入数list]
    # condition.1 : a点以上のブランドがb社以上
    result_condition1 = []
    for i in range(len(list_in_list)):
        ith_id = list_in_list[i].pop(0)
        condition1_list = [ x for x in list_in_list[i] if int(x) >= a ]
        if len(condition1_list) >= b:
            list_in_list[i].insert(0,ith_id)
            result_condition1.append(list_in_list[i])
    # condition.2 : c点以上のブランドがある
    result_condition2 = []
    for i in range(len(result_condition1)):
        ith_id = result_condition1[i].pop(0)
        condition2_list = [ x for x in result_condition1[i] if int(x) >= c ]
        if len(condition2_list) != 0:
            result_condition1[i].insert(0,ith_id)
            result_condition2.append(result_condition1[i])
    # condition.3 : 総数d点以下
    result_condition3 = []
    for i in range(len(result_condition2)):
        ith_id = result_condition2[i].pop(0)
        int_list = [ int(n) for n in result_condition2[i] ]
        if sum(int_list) <= d:
            int_list.insert(0,int(ith_id))
            result_condition3.append(int_list)
    # condition.4 : 上位condition_brands位まで選抜
    ## brandごとの購入数 all_purchase_list = [[0,b1_purchase],[1,b2_purchase],...,[num_brands-1,b(num_brands)_purchase]]
    ## (※条件を満たすplayerのデータのみの合計．元の実データではないので注意．)
    result_condition4 = []
    all_purchase_list = [ [i,0] for i in range(num_brands) ]
    for i in range(len(result_condition3)):
        for j in range(num_brands):
            if j != 0:
                all_purchase_list[j-1][1] += result_condition3[i][j]
    sort_purchase_list = sorted(all_purchase_list,key=itemgetter(1),reverse=True)
    purchase_list = [ sort_purchase_list[i] for i in range(condition_brands) ]
    result_condition4_id = [ int(sort_purchase_list[i][0]) for i in range(condition_brands) ]
    result_condition4_id.sort()
    for i in range(len(result_condition3)):
        append_list = []
        for j in range(num_brands+1):
            if j == 0:
                append_list.append(result_condition3[i][0])
            elif int(j-1) in result_condition4_id:
                append_list.append(result_condition3[i][j])
        result_condition4.append(append_list)
    return [result_condition4,result_condition4_id,all_purchase_list,purchase_list]

def ranking(list_in_list,fil,condition_brands,directory,fullpath): # return -> [ranking,fullranking]
    # id & ranking $ fullranking
    result_ranking = []
    result_fullranking = []
    exp_result_ranking = []
    exp_result_fullranking = []
    # fullranking
    for i in range(len(list_in_list)):
        remove_id = int(list_in_list[i].pop(0))
        set_list = [ [j,list_in_list[i][j]] for j in range(len(list_in_list[i])) if int(list_in_list[i][j]) != 0 ]
        sort_list = sorted(set_list, key = lambda x : x[1], reverse = True)
        ## すべてを購入していなくてもok
        ### fullranking
        append_fullranking = [ remove_id ]
        for j in range(len(sort_list)):
            append_fullranking.append(int(sort_list[j][0]))
        if len(append_fullranking) >= 3:
            result_fullranking.append(append_fullranking)
        ### ranking
        ith_brand = list(range(len(sort_list))) # brand_idの位置
        ith_brand_comb = list(itertools.combinations(ith_brand, 2)) # 組み合わせ
        for j in range(len(ith_brand_comb)):
            ### 2ブランドを比較する際に一方が他方のfil倍以上のときだけを抽出
            if float(sort_list[ith_brand_comb[j][0]][1]) >= float(sort_list[ith_brand_comb[j][1]][1]) * float(fil):
                result_ranking.append([remove_id , int(sort_list[ith_brand_comb[j][0]][0]) , int(sort_list[ith_brand_comb[j][1]][0]) ])
        ## confition_brandsすべてを購入しているplayerのみを選抜 (実験用)/////////////////////////////////////
        explist_1 = [ int(sort_list[j][0]) for j in range(len(sort_list)) ]
        if len(explist_1) == int(condition_brands):
            exp_result_fullranking.append(explist_1)
        ## ////////////////////////////////////////////////////////////////////////////////////////////
    # ranking
    exp_brand_comb = list(itertools.combinations(range(condition_brands),2)) # 組み合わせ        
    for j in range(len(exp_result_fullranking)):
        for k in range(len(exp_brand_comb)):
            exp_result_ranking.append([j,exp_result_fullranking[j][exp_brand_comb[k][0]],exp_result_fullranking[j][exp_brand_comb[k][1]]])
    print(len(exp_result_ranking)) # [TEST]
    # id re-numbering
    ## fullranking
    for i in range(len(result_fullranking)):
        result_fullranking[i][0] = i
    ## ranking
    result_ranking_csv = []
    for i in range(len(result_ranking)):
        if i == 0:
            result_ranking_csv.append([0,int(result_ranking[i][1]),int(result_ranking[i][2])])
        else:
            if result_ranking[i][0] == result_ranking[i-1][0]:
                result_ranking_csv.append([int(result_ranking[i-1][0]),int(result_ranking[i][1]),int(result_ranking[i][2])])
            else:
                result_ranking_csv.append([int(result_ranking[i-1][0])+1,int(result_ranking[i][1]),int(result_ranking[i][2])])
    # Save csvfile
    ## fullranking.csv
    if not os.path.exists(directory+'/fullranking.csv'):
        csv_save(result_fullranking,directory+'/fullranking.csv')
    ## ranking.csv
    if not os.path.exists(directory+'/ranking.csv'):
        csv_save(result_ranking_csv,directory+'/ranking.csv')
    ## 実験用 //////////////////////////////////////////////////////////////////////////////////////////
    ### exp_fullranking.csv
    if not os.path.exists(fullpath+'/actual_data/'+str(condition_brands)+'_'+str(len(exp_result_fullranking))+'/full_ranking.csv'):
        os.makedirs(fullpath+'/actual_data/'+str(condition_brands)+'_'+str(len(exp_result_fullranking)))
        csv_save(exp_result_fullranking,fullpath+'/actual_data/'+str(condition_brands)+'_'+str(len(exp_result_fullranking))+'/full_ranking.csv')
    ### exp_ranking.csv
    if not os.path.exists(fullpath+'/actual_data/'+str(condition_brands)+'_'+str(len(exp_result_fullranking))+'_actual/ranking.csv'):
        csv_save(exp_result_ranking,fullpath+'/actual_data/'+str(condition_brands)+'_'+str(len(exp_result_fullranking))+'/ranking.csv')
    ## ////////////////////////////////////////////////////////////////////////////////////////////////
    return [result_ranking_csv,result_fullranking]

def distance(list1,list2):
    d = 0
    for i in range(len(list1)):
        d += (list1[i]-list2[i]) ** 2
    d = math.sqrt(d)
    return d

def area(point_list,delta,condition_brands):
    result = [0 for i in range(condition_brands)]
    coordinates = [-1+2*(i/delta) for i in range(delta+1)] #[-1,1]をdelta分割
    num_points = (delta+1)**2
    for i,j in itertools.product(coordinates,coordinates):
        point = [i,j]
        distance_list = []
        for k in range(len(point_list)):
            distance_list.append([k,distance(point,point_list[k])])
        distance_list.sort(key=itemgetter(1))
        nearest_brand = distance_list[0][0]
        result[int(nearest_brand)] += 1
    return result

def log_open(pathcode): # return -> [str_list(保存用),loss_ord_list]
    f = open(str(pathcode)+'/log')
    flines = f.readlines()
    log_list = [s for s in flines if 'opt/loss_ord' in s]
    cut_list = [re.findall(r'[0-9]+',log_list[i]) for i in range(len(log_list))]
    str_list = []
    for i in range(len(cut_list)):
        if len(cut_list[i]) == 3:
            str_list.append(cut_list[i][0]+'.'+cut_list[i][1]+'e-'+cut_list[i][2])
        elif len(cut_list[i]) == 2:
            str_list.append('.'.join(cut_list[i]))
        else:
            str_list.append(cut_list[i])
    loss_ord_list = list(map(float,str_list))
    f.close()
    return [str_list,loss_ord_list]

def correlation(list1,list2):
    s1 = pd.Series(list1)
    s2 = pd.Series(list2)
    res = s1.corr(s2)
    return res

def evaluate(fullranking,result_brands_points,condition_brands,delta,directory,rp,rb): # return -> [ranking_count,result(cor list)]
    # rankingの総順列list brands_permutation の作成
    list_1 = [str(i) for i in range(condition_brands)]
    list_2 = list(itertools.permutations(list_1))
    brands_permutation = [ ''.join(list_2[i]) for i in range(len(list_2)) ]
    permu_terms = len(list_2)
    # ranking_count の作成
    ranking_count = [ [brands_permutation[i],0] for i in range(len(brands_permutation)) ]
    ## fullranking のカウント
    for i in range(len(fullranking)):
        fullranking[i].pop(0)
        if len(fullranking[i]) == condition_brands:
            list_i = list(map(str,fullranking[i]))
            element = ''.join(list_i)
            if element in brands_permutation:
                ranking_count[int(brands_permutation.index(element))][1] += 1
            else:
                print("ERROR1. function evaluate")
    # result_ranking_count の作成
    result_ranking_count = [ [brands_permutation[i],0] for i in range(len(brands_permutation)) ]
    ## 各点(points)の距離順列
    coordinates = [-1+2*(i/delta) for i in range(delta+1)] #[-1,1]をdelta分割
    num_points = (delta+1)**2
    for i,j in itertools.product(coordinates,coordinates):
        point = [i,j]
        distance_list = []
        for k in range(len(result_brands_points)):
            distance_list.append([k,distance(point,result_brands_points[k])])
        distance_list.sort(key=itemgetter(1))
        distance_rank = [ str(distance_list[i][0]) for i in range(len(distance_list)) ]
        result_element = ''.join(distance_rank)
        if result_element in brands_permutation:
            result_ranking_count[int(brands_permutation.index(result_element))][1] += 1
        else:
            print("ERROR2. function evaluate")
    # 相関係数の集計
    result = []
    ## n-iブランド -> 残りiブランド (2<=i<n, i=2,...,n-1) を推定
    for i in range(2,condition_brands):
        rep = math.factorial(i)
        term = int(permu_terms / rep)
        ith_result = [i,[]]
        ### 比較listの作成 & 相関係数の計算 & ith_result[1]への追加
        for j in range(term):
            truth_check = [ int(ranking_count[int(j*rep+k)][1]) for k in range(rep) ]
            result_check = [ int(result_ranking_count[int(j*rep+k)][1]) for k in range(rep) ]
            ith_jth_cor = float(correlation(truth_check,result_check))
            ith_result[1].append(ith_jth_cor)
        ### resultへの追加
        result.append(ith_result)
    # Save csvfile
    ## ranking_count.csv
    if not os.path.exists(directory+'/ranking_count.csv'):
        csv_save(ranking_count,directory+'/ranking_count.csv')
    #////////////////////////////////////////////////////////////////////////////////////
    ## result_ranking_count.csv
    if not os.path.exists(directory+'/result/rprb_'+str(rp)+'_'+str(rb)+'/result_ranking_count.csv'):
        csv_save(result_ranking_count,directory+'/result/rprb_'+str(rp)+'_'+str(rb)+'/result_ranking_count.csv')
    #////////////////////////////////////////////////////////////////////////////////////
    ## evaluate.csv
    if not os.path.exists(directory+'/result/rprb_'+str(rp)+'_'+str(rb)+'/evaluate.csv'):
        csv_save(result,directory+'/result/rprb_'+str(rp)+'_'+str(rb)+'/evaluate.csv')
    return [ranking_count,result]

if __name__ == '__main__':
    # input value
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help='zozo.csv(実データ)が格納されているディレクトリ', default='./actual_data')
    parser.add_argument('-fp', '--fullpath', type=str, help='fullranking実験を格納するディレクトリ', default='./0422_players')
    parser.add_argument('-ma', '--min_value_a', type=int, help='condition.1 購入数(a)の最小値', default=2)
    parser.add_argument('-na', '--num_a', type=int, help='aの実験個数(ma,ma+1,...,ma+(na-1))', default=1)
    parser.add_argument('-mb', '--min_value_b', type=int, help='condition.1 対象ブランド社数(b)の最小値', default=2)
    parser.add_argument('-nb', '--num_b', type=int, help='bの実験個数(mb,mb+1,...,mb+(nb-1))', default=1)
    parser.add_argument('-mc', '--min_value_c', type=int, help='condition.2 購入数(c)の最小値', default=4)
    parser.add_argument('-nc', '--num_c', type=int, help='cの実験個数(mc,mc+1,...,mc+(nc-1))', default=1)
    parser.add_argument('-d', type=int, help='condition.3 総数d点以下', default=700)
    parser.add_argument('-f', type=float, help='ブランド購入数の比較時の倍率', default=1.0)
    parser.add_argument('-mrp', '--min_rp', type=float, help='-rpの最小値', default=1e-9)
    parser.add_argument('-nrp', '--num_rp', type=int, help='-rpの実験個数', default=1)
    parser.add_argument('-mrb', '--min_rb', type=float, help='-rbの最小値', default=0.01)
    parser.add_argument('-nrb', '--num_rb', type=int, help='-rbの実験個数', default=1)
    parser.add_argument('-e', '--epoch', type=int, help='実験epoch数', default=100)
    parser.add_argument('-n', type=int, help='並列数', default=4)
    parser.add_argument('-del', '--delta', type=int, help='面積推定時の座標分割数', default=400)
    parser.add_argument('-bra', '--condition_brands', type=int, help='選抜する上位ブランドb社')
    parser.add_argument('-fe', '--fullranking_exp', action='store_true', help='players_exp.py実行時にa,b,c値によるデータ選抜のみか？')
    args = parser.parse_args()

    # def of list and const.
    lista = [ args.min_value_a + i for i in range(args.num_a) ] # lista = [ma,ma+1,...,ma+(na-1)]
    listb = [ args.min_value_b + i for i in range(args.num_b) ]
    listc = [ args.min_value_c + i for i in range(args.num_c) ]
    listrp = [ float(args.min_rp * (10 ** i)) for i in range(args.num_rp) ] # listrp = [mrp,mrp*10,...,mrp*10^(nrp-1)]
    listrb = [ float(args.min_rb * (10 ** i)) for i in range(args.num_rb) ]
    d = args.d
    path = args.path

    # read zozo.csv
    original = csv_open(path+'/zozo.csv')
    
    # main
    for a,b,c,rp,rb in tqdm(itertools.product(lista,listb,listc,listrp,listrb)):
        # make dir
        ## arrangement
        step1_1 = arrangement(original)
        list_step1_1 = step1_1[0]
        ## 選抜ブランド数 condition_brands の定義
        num_brands = step1_1[2]
        if args.condition_brands:
            condition_brands = args.condition_brands
        else:
            condition_brands = num_brands
        ## make dir
        directory = str(path+'/brand_'+str(condition_brands)+'/'+str(a)+'_'+str(b)+'_'+str(c)+'_'+str(d)+'_'+str(args.f))
        if not os.path.exists(directory):
            os.makedirs(directory+'/result')

        # ranking.csv の作成 or 読み込み
        ## condition check
        step1_2 = condition(list_step1_1,a,b,c,d,num_brands,condition_brands)
        list_step1_2 = step1_2[0]
        condition_brands_id = step1_2[1]
        all_purchase_list = [ step1_2[2][i][1] for i in range(len(step1_2[2])) ] # 全ブランドの購入数のみ(idなし)
        purchase_list = [ step1_2[3][i][1] for i in range(len(step1_2[3])) ] # 選抜ブランドの購入数のみ(idなし)
        ## ranking
        step1_3 = ranking(list_step1_2,args.f,condition_brands,directory,args.fullpath)
        ### players_exp.py 実行時はここで脱出
        if args.fullranking_exp:
            continue
        ### actual_exp.py 実行時は続行
        ranking_list = step1_3[0]
        fullranking = step1_3[1]
        num_data = len(ranking_list)
        ## print
        print("data数:",num_data)

        # すでに先行結果が存在する -> 結果を読み込み
        # 先行結果が存在しない -> newrankLearn.pyの実行 (epoch=100)
        if os.path.exists(directory+'/result/rprb_'+str(rp)+'_'+str(rb)+'/out_brands0000.csv'):
            ## Open out_brands.csv
            result_brands_str = csv_open(directory+'/result/rprb_'+str(rp)+'_'+str(rb)+'/out_brands0000.csv')
            ### Open log
            save_lossord = log_open(directory+'/result/rprb_'+str(rp)+'_'+str(rb))[0]
        else:
            ## file名(日付_時刻)の取得
            dt = datetime.datetime.now()
            filename = str('{0:02d}'.format(dt.month)+'{0:02d}'.format(dt.day)+'_'+'{0:02d}'.format(dt.hour)+'{0:02d}'.format(dt.minute))
            if dt.minute == 59:
                filename2 = str('{0:02d}'.format(dt.month)+'{0:02d}'.format(dt.day)+'_'+'{0:02d}'.format(dt.hour+1)+'00')
            else:
                filename2 = str('{0:02d}'.format(dt.month)+'{0:02d}'.format(dt.day)+'_'+'{0:02d}'.format(dt.hour)+'{0:02d}'.format(dt.minute+1))
            ## newrankLearn.py の実行
            os.system(
                'mpiexec -n '+str(args.n)+' python newrankLearn.py '+str(directory)+'/ranking.csv -e '+str(int(args.epoch/args.n))+' -rp '+str(rp)+' -rb '+str(rb)+' -o '+str(directory+'/result')+' --mpi'
            )
            ## Open out_brands.csv
            if os.path.exists(directory+'/result/'+filename):
                ### Open out_brands.csv
                result_brands_str = csv_open(directory+'/result/'+filename+'/out_brands0000.csv')
                ### Open log
                save_lossord = log_open(directory+'/result/'+filename)[0]
                ### ファイル名の整理 date_time -> rp_rb
                os.rename(directory+'/result/'+filename, directory+'/result/rprb_'+str(rp)+'_'+str(rb))
            elif os.path.exists(directory+'/result/'+filename2):
                ### Open out_brands.csv
                result_brands_str = csv_open(directory+'/result/'+filename2+'/out_brands0000.csv')
                ### Open log
                save_lossord = log_open(directory+'/result/'+filename2)[0]
                ### ファイル名の整理 date_time -> rp_rb
                os.rename(directory+'/result/'+filename2, directory+'/result/rprb_'+str(rp)+'_'+str(rb))
        ## result_brands_points の作成
        result_brands_points = []
        for i in range(len(result_brands_str)):
            result_brands_points.append(list(map(float, result_brands_str[i])))

        # 評価ver.1 (領域面積)
        ## result_brands_points から各ブランドの領域面積を推定 -> result1_list
        result1_list = area(result_brands_points,args.delta,condition_brands)
        ## 相関係数計算
        cor1 = correlation(purchase_list,result1_list)
        print("領域面積比較での決定率:",cor1)

        # 評価ver.2（3ブランドの購入数から残り2ブランドの面積比を導出）
        eva2 = evaluate(fullranking,result_brands_points,condition_brands,args.delta,directory,rp,rb)
        eva2_result = eva2[1]

        # Save data
        ## summary.csv
        ### exp_id
        summary_data = csv_open(path+'/summary.csv')
        exp_id = int(len(summary_data)-1)
        ### this exp's data
        data_list = [exp_id,condition_brands,a,b,c,d,args.f,rp,rb,num_data,args.epoch,cor1,condition_brands_id]
        ### new summary_data
        summary_data.append(data_list)
        ### Save
        csv_save(summary_data,path+'/summary.csv')
        ## ./loss_ord_log/loss_ord_exp_id.csv
        lossord_savelist = [[x] for x in save_lossord]
        csv_save(lossord_savelist,path+'/loss_ord_log/loss_ord_'+str(exp_id)+'.csv')
            








        










