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
#import re
#import copy
#import sys
from tqdm import tqdm
#from operator import itemgetter
#import pprint
import itertools
import argparse
import time
#import statistics
from scipy.spatial.distance import cdist

# data -> csvfile
def csvsave(data,path,savetype):
    # ready
    if type(data) == list:
        savedata = np.array(data)
    elif type(data) == int or type(data) == float:
        savedata = np.array([data])
    elif type(data) == str:
        savedata = np.array([float(data)])
    elif type(data).__module__ == np.__name__:
        savedata = data
    # save
    np.savetxt(path,savedata,delimiter=',',fmt=str(savetype))

# csvfile -> data
def csvopen(path,opentype):
    return np.loadtxt(path,delimiter=',',dtype=str(opentype))

# sushi3a.5000.10.order -> 嗜好順データ5000人分だけを行列化
def datamatrix(args):
    datapath = args.path + '/data/original/sushi3a.5000.10.order'
    with open(datapath) as file:
            reader = csv.reader(file)
            list1 = [row for row in reader]
            list1.pop(0)
    list2 = []
    for i in range(len(list1)):
        list3 = list1[i][0].split()
        for j in range(len(list3)):
            list3[j] = int(list3[j])
        list2.append(list3)
    matrix = np.delete(np.array(list2),[0,1],1)
    return matrix

# datamatrixから実験対象となる銘柄数だけ選抜
def select(datamatrix,args):
    # const
    numbrands = len(args.selectbrands)
    # do
    resultmatrix = np.empty((datamatrix.shape[0],numbrands))
    for i in range(datamatrix.shape[0]):
        vec = datamatrix[i,:]
        resultmatrix[i,:] = [ int(args.selectbrands.index(vec[j])) for j in range(len(vec)) if vec[j] in args.selectbrands ]
    return resultmatrix

# fullranking -> ranking
def rankingmatrix(datamatrix):
    # const
    numbrands = datamatrix.shape[1]
    numplayers = datamatrix.shape[0]
    list1 = []
    # do
    for i in range(numplayers):
        fullvec = datamatrix[i,:]
        combination = list(itertools.combinations(list(range(numbrands)),2))
        for j in range(len(combination)):
            list1.append([i,fullvec[combination[j][0]],fullvec[combination[j][1]]])
    rankingmatrix = np.array(list1,dtype='int')
    return rankingmatrix

# new rankLearning.py
def rankLearning(datapath,args):
    dt = datetime.datetime.now()
    ## def of filename & filename2
    filename = str('{0:02d}'.format(dt.month)+'{0:02d}'.format(dt.day)+'_'+'{0:02d}'.format(dt.hour)+'{0:02d}'.format(dt.minute))
    if dt.minute == 59:
        filename2 = str('{0:02d}'.format(dt.month)+'{0:02d}'.format(dt.day)+'_'+'{0:02d}'.format(dt.hour+1)+'00')
    else:
        filename2 = str('{0:02d}'.format(dt.month)+'{0:02d}'.format(dt.day)+'_'+'{0:02d}'.format(dt.hour)+'{0:02d}'.format(dt.minute+1))
    ## newrankLearn.py
    os.system(
        'mpiexec -n '+str(args.n)+' python newrankLearn.py '+datapath+'/ranking.csv -e '+str(int(args.epoch/args.n))+' -rp '+str(args.rp)+' -rb '+str(args.rb)+' -d '+str(args.dim)+' -o '+str(resultpath)+' --mpi'
    )
    ## filename change -> './no_?/--brandids--/rprb_??_??'
    if os.path.exists(resultpath+'/'+filename):
        os.rename(resultpath+'/'+filename,resultpath+'/rprb_'+str(args.rp)+'_'+str(args.rb))
    elif os.path.exists(resultpath+'/'+filename2):
        os.rename(resultpath+'/'+filename2,resultpath+'/rprb_'+str(args.rp)+'_'+str(args.rb))

# fullranking -> area計算 (evaluate関数用)
def area(fullmatrix,args,checknb,rankno):
    # const
    #numbrands = len(args.selectbrands)
    numplayers = fullmatrix.shape[0]
    x = 0
    # permutation
    selectrank = list(itertools.combinations(range(len(args.selectbrands)),checknb))[rankno]
    ranktuple = list(itertools.permutations(selectrank))
    rankmatrix = np.array([ list(ranktuple[i]) for i in range(len(ranktuple)) ])
    # do
    result = np.array([0 for i in range(rankmatrix.shape[0])])
    print('area function...')
    for i in tqdm(range(rankmatrix.shape[0])):
        for j in range(x,numplayers):
            if np.all(fullmatrix[j,:] == rankmatrix[i,:]):
                result[i] += 1
                x += 1
            else:
                break
    return result

# fullranking -> area計算 (topevaluate関数用)
def toparea(fullmatrix,args):
    # const
    #numbrands = len(args.selectbrands)
    numplayers = fullmatrix.shape[0]
    x = 0
    # permutation
    ranktuple = list(itertools.permutations(range(len(args.selectbrands)),args.topnumbrands))
    rankmatrix = np.array([ list(ranktuple[i]) for i in range(len(ranktuple)) ])
    # do
    result = np.array([0 for i in range(rankmatrix.shape[0])])
    print('area function...')
    for i in tqdm(range(rankmatrix.shape[0])):
        for j in range(x,numplayers):
            if np.all(fullmatrix[j,:] == rankmatrix[i,:]):
                result[i] += 1
                x += 1
            else:
                break
    return result

# fullrankingをsortする
def ranksort(fullmatrix,args,checknb):
    #numbrands = len(args.selectbrands)
    fulllist = fullmatrix.tolist()
    print('ranksort function...')
    for i in tqdm(range(checknb)):
        fulllist.sort(key=lambda x: x[int(-1*(i+1))])
    result = np.array(fulllist)
    return result

# truth area & result area & correlation coefficientの計算
"""
def evaluate(resultpoint,truthfullmatrix,args,evaselectbrands):
    # const.
    numbrands = len(args.selectbrands)
    # distance matrix
    randompoint = 2*np.random.rand(args.u,args.dim)-1
    distancematrix = cdist(randompoint,resultpoint)
    # resultfullranking -> rmatrix -> resultmatrix
    evaluatefile = args.path+'/result/no_'+str(args.expno)+'/'+str(csvdataname)+'/dim_'+str(args.dim)+'/rprb_'+str(args.rp)+'_'+str(args.rb)+'/evaluate_'+str(evaselectbrands)+'/evaluate.csv'
    if os.path.isfile(evaluatefile):
        rmatrix = csvopen(evaluatefile,'int')
    else:
        rmatrix = np.empty((args.u,numbrands),dtype='int64')
        print('evaluate function...') # [TEST!]
        for i in tqdm(range(args.u)):
            sort = [ np.argsort(distancematrix[i,:])[j] for j in range(numbrands) ]
            rmatrix[i,:] = np.array(sort)
        csvsave(rmatrix,evaluatefile,'%d')
    resultmatrix = ranksort(rmatrix,args)
    # truthfullranking -> truthmatrix
    truthmatrix = ranksort(truthfullmatrix,args)
    # area calculation
    truth = area(truthmatrix,args)
    result = area(resultmatrix,args)
    # correlation coefficient
    s1 = pd.Series(truth)
    s2 = pd.Series(result)
    cor = s1.corr(s2)
    return [truth,result,cor]
"""

# 評価に用いるbrand数の組み合わせに選抜する
def selectevaluate(fullmatrix,args,checknb):
    # set
    selecttuple = list(itertools.combinations(range(len(args.selectbrands)),checknb))
    selectmatrix = np.array([ list(selecttuple[i]) for i in range(len(selecttuple)) ])
    result = np.empty((selectmatrix.shape[0],fullmatrix.shape[0],checknb),dtype='int64')
    # do
    print('selecting in evaluation...')
    for i in tqdm(range(selectmatrix.shape[0])):
        selectlist = selectmatrix[i,:].tolist()
        interim = np.empty((fullmatrix.shape[0],checknb),dtype='int64')
        for j in range(fullmatrix.shape[0]):
            player = fullmatrix[j,:].tolist()
            new = [ player[k] for k in range(len(player)) if player[k] in selectlist ]
            interim[j,:] = np.array(new,dtype='int64')
        result[i,:,:] = interim
    return result

# truth area & result area & correlation coefficientの計算
def evaluate(resultpoint,truthfullmatrix,args,checknb):
    # const. (numbrands -> 学習させたbrand数 ／ checknb -> 評価で用いるbrand数)
    numbrands = len(args.selectbrands)
    # distance matrix
    randompoint = 2*np.random.rand(args.u,args.dim)-1
    distancematrix = cdist(randompoint,resultpoint)
    # resultfullranking -> rmatrix (sortなし)
    evaluatefile = args.path+'/result/no_'+str(args.expno)+'/'+str(csvdataname)+'/dim_'+str(args.dim)+'/rprb_'+str(args.rp)+'_'+str(args.rb)+'/evaluate.csv'
    os.makedirs(args.path+'/result/no_'+str(args.expno)+'/'+str(csvdataname)+'/dim_'+str(args.dim)+'/rprb_'+str(args.rp)+'_'+str(args.rb)+'/evaluate_'+str(checknb),exist_ok=True)
    if os.path.isfile(evaluatefile):
        rmatrix = csvopen(evaluatefile,'int')
    else:
        rmatrix = np.empty((args.u,numbrands),dtype='int64')
        print('evaluate function...') # [TEST!]
        for i in tqdm(range(args.u)):
            sort = [ np.argsort(distancematrix[i,:])[j] for j in range(numbrands) ]
            rmatrix[i,:] = np.array(sort)
        ## save
        csvsave(rmatrix,evaluatefile,'%d')
    # selection [ new! ]
    selectedrmatrixfile = args.path+'/result/no_'+str(args.expno)+'/'+str(csvdataname)+'/dim_'+str(args.dim)+'/rprb_'+str(args.rp)+'_'+str(args.rb)+'/evaluate_'+str(checknb)+'/rmatrix.npy'
    selectedtmatrixfile = args.path+'/result/no_'+str(args.expno)+'/'+str(csvdataname)+'/dim_'+str(args.dim)+'/rprb_'+str(args.rp)+'_'+str(args.rb)+'/evaluate_'+str(checknb)+'/tmatrix.npy'
    numcomb = len(list(itertools.combinations(range(len(args.selectbrands)),checknb)))
    if os.path.isfile(selectedrmatrixfile) and os.path.isfile(selectedtmatrixfile):
        selectedrmatrix = np.load(selectedrmatrixfile)
        selectedtmatrix = np.load(selectedtmatrixfile)
    else:
        ## selection
        selectedrmatrix = selectevaluate(rmatrix,args,checknb)
        selectedtmatrix = selectevaluate(truthfullmatrix,args,checknb)
        ## save
        np.save(selectedrmatrixfile,selectedrmatrix)
        np.save(selectedtmatrixfile,selectedtmatrix)
    # sort (truth&result)
    resultmatrixfile = args.path+'/result/no_'+str(args.expno)+'/'+str(csvdataname)+'/dim_'+str(args.dim)+'/rprb_'+str(args.rp)+'_'+str(args.rb)+'/evaluate_'+str(checknb)+'/resultmatrix.npy'
    truthmatrixfile = args.path+'/result/no_'+str(args.expno)+'/'+str(csvdataname)+'/dim_'+str(args.dim)+'/rprb_'+str(args.rp)+'_'+str(args.rb)+'/evaluate_'+str(checknb)+'/truthmatrix.npy'
    if os.path.isfile(resultmatrixfile) and os.path.isfile(truthmatrixfile):
        resultmatrix = np.load(resultmatrixfile)
        truthmatrix = np.load(truthmatrixfile)
    else:
        resultmatrix = np.empty((selectedrmatrix.shape[0],selectedrmatrix.shape[1],selectedrmatrix.shape[2]),dtype='int64')
        truthmatrix = np.empty((selectedtmatrix.shape[0],selectedtmatrix.shape[1],selectedtmatrix.shape[2]),dtype='int64')
        for i in range(numcomb):
            ## result
            resultmatrix[i,:,:] = ranksort(selectedrmatrix[i,:,:],args,checknb)
            ## truth
            truthmatrix[i,:,:] = ranksort(selectedtmatrix[i,:,:],args,checknb)
        ## save
        np.save(resultmatrixfile,resultmatrix)
        np.save(truthmatrixfile,truthmatrix)
    # area calculation
    resultfile = args.path+'/result/no_'+str(args.expno)+'/'+str(csvdataname)+'/dim_'+str(args.dim)+'/rprb_'+str(args.rp)+'_'+str(args.rb)+'/evaluate_'+str(checknb)+'/result.csv'
    truthfile = args.path+'/result/no_'+str(args.expno)+'/'+str(csvdataname)+'/dim_'+str(args.dim)+'/rprb_'+str(args.rp)+'_'+str(args.rb)+'/evaluate_'+str(checknb)+'/truth.csv'
    if os.path.isfile(resultfile) and os.path.isfile(truthfile):
        result = csvopen(resultfile,'int')
        truth = csvopen(truthfile,'int')
    else:
        result = np.empty((numcomb,len(list(itertools.permutations(list(range(checknb)))))),dtype='int64')
        truth = np.empty((numcomb,len(list(itertools.permutations(list(range(checknb)))))),dtype='int64')
        ############################################
        for i in range(numcomb):
            result[i,:] = area(resultmatrix[i,:,:],args,checknb,i)
            truth[i,:] = area(truthmatrix[i,:,:],args,checknb,i)
        ## save
        csvsave(result,resultfile,'%d')
        csvsave(truth,truthfile,'%d')
    # correlation coefficient
    corfile = args.path+'/result/no_'+str(args.expno)+'/'+str(csvdataname)+'/dim_'+str(args.dim)+'/rprb_'+str(args.rp)+'_'+str(args.rb)+'/evaluate_'+str(checknb)+'/cor.csv'
    if os.path.isfile(corfile):
        cor = csvopen(corfile,'float')
    else:
        cor = np.empty(numcomb)
        for i in range(numcomb):
            s1 = pd.Series(truth[i,:])
            s2 = pd.Series(result[i,:])
            cor[i] = s1.corr(s2)
        ## save
        csvsave(cor,corfile,'%f')
        ## delete noneed files
        os.remove(selectedrmatrixfile)
        os.remove(selectedtmatrixfile)
        os.remove(resultmatrixfile)
        os.remove(truthmatrixfile)
    return [truth,result,cor]

# truth area & result area & correlation coefficientの計算
def topevaluate(resultpoint,truthfullmatrix,args):
    # const.
    numbrands = len(args.selectbrands)
    # distance matrix
    randompoint = 2*np.random.rand(args.u,args.dim)-1
    distancematrix = cdist(randompoint,resultpoint)
    # resultfullranking -> rmatrix (sortなし)
    evaluatefile = args.path+'/result/no_'+str(args.expno)+'/'+str(csvdataname)+'/dim_'+str(args.dim)+'/rprb_'+str(args.rp)+'_'+str(args.rb)+'/evaluate.csv'
    os.makedirs(args.path+'/result/no_'+str(args.expno)+'/'+str(csvdataname)+'/dim_'+str(args.dim)+'/rprb_'+str(args.rp)+'_'+str(args.rb)+'/top_'+str(args.topnumbrands),exist_ok=True)
    if os.path.isfile(evaluatefile):
        rmatrix = csvopen(evaluatefile,'int')
    else:
        rmatrix = np.empty((args.u,numbrands),dtype='int64')
        print('evaluate function...') # [TEST!]
        for i in tqdm(range(args.u)):
            sort = [ np.argsort(distancematrix[i,:])[j] for j in range(numbrands) ]
            rmatrix[i,:] = np.array(sort)
        ## save
        csvsave(rmatrix,evaluatefile,'%d')
    # top selection
    topselectedrmatrixfile = args.path+'/result/no_'+str(args.expno)+'/'+str(csvdataname)+'/dim_'+str(args.dim)+'/rprb_'+str(args.rp)+'_'+str(args.rb)+'/top_'+str(args.topnumbrands)+'/rmatrix.npy'
    topselectedtmatrixfile = args.path+'/result/no_'+str(args.expno)+'/'+str(csvdataname)+'/dim_'+str(args.dim)+'/rprb_'+str(args.rp)+'_'+str(args.rb)+'/top_'+str(args.topnumbrands)+'/tmatrix.npy'
    if os.path.isfile(topselectedrmatrixfile) and os.path.isfile(topselectedtmatrixfile):
        topselectedrmatrix = np.load(topselectedrmatrixfile)
        topselectedtmatrix = np.load(topselectedtmatrixfile)
    else:
        ## selection
        topselectedrmatrix = np.delete(rmatrix,list(range(args.topnumbrands,numbrands)),1)
        topselectedtmatrix = np.delete(truthfullmatrix,list(range(args.topnumbrands,numbrands)),1)
        ## save
        np.save(topselectedrmatrixfile,topselectedrmatrix)
        np.save(topselectedtmatrixfile,topselectedtmatrix)
    # sort (truth&result)
    topresultmatrixfile = args.path+'/result/no_'+str(args.expno)+'/'+str(csvdataname)+'/dim_'+str(args.dim)+'/rprb_'+str(args.rp)+'_'+str(args.rb)+'/top_'+str(args.topnumbrands)+'/resultmatrix.npy'
    toptruthmatrixfile = args.path+'/result/no_'+str(args.expno)+'/'+str(csvdataname)+'/dim_'+str(args.dim)+'/rprb_'+str(args.rp)+'_'+str(args.rb)+'/top_'+str(args.topnumbrands)+'/truthmatrix.npy'
    if os.path.isfile(topresultmatrixfile) and os.path.isfile(toptruthmatrixfile):
        topresultmatrix = np.load(topresultmatrixfile)
        toptruthmatrix = np.load(toptruthmatrixfile)
    else:
        topresultmatrix = ranksort(topselectedrmatrix,args,args.topnumbrands)
        toptruthmatrix = ranksort(topselectedtmatrix,args,args.topnumbrands)
        ## save
        np.save(topresultmatrixfile,topresultmatrix)
        np.save(toptruthmatrixfile,toptruthmatrix)
    # area calculation
    topresultfile = args.path+'/result/no_'+str(args.expno)+'/'+str(csvdataname)+'/dim_'+str(args.dim)+'/rprb_'+str(args.rp)+'_'+str(args.rb)+'/top_'+str(args.topnumbrands)+'/result.csv'
    toptruthfile = args.path+'/result/no_'+str(args.expno)+'/'+str(csvdataname)+'/dim_'+str(args.dim)+'/rprb_'+str(args.rp)+'_'+str(args.rb)+'/top_'+str(args.topnumbrands)+'/truth.csv'
    if os.path.isfile(topresultfile) and os.path.isfile(toptruthfile):
        topresult = csvopen(topresultfile,'int')
        toptruth = csvopen(toptruthfile,'int')
    else:
        topresult = toparea(topresultmatrix,args)
        toptruth = toparea(toptruthmatrix,args)
        ## save
        csvsave(topresult,topresultfile,'%d')
        csvsave(toptruth,toptruthfile,'%d')
    # correlation coefficient
    topcorfile = args.path+'/result/no_'+str(args.expno)+'/'+str(csvdataname)+'/dim_'+str(args.dim)+'/rprb_'+str(args.rp)+'_'+str(args.rb)+'/top_'+str(args.topnumbrands)+'/cor.csv'
    if os.path.isfile(topcorfile):
        topcor = csvopen(topcorfile,'float')
    else:
        s1 = pd.Series(toptruth)
        s2 = pd.Series(topresult)
        topcor = np.array([s1.corr(s2)])
        ## save
        csvsave(topcor,topcorfile,'%f')
        ## delete noneed files
        os.remove(topselectedrmatrixfile)
        os.remove(topselectedtmatrixfile)
        os.remove(topresultmatrixfile)
        os.remove(toptruthmatrixfile)
    return [toptruth,topresult,topcor]

# 上位N位までに区切ったら何%はうまくできてるの？
## rankLearning
def faithrankLearning(datapath,faithresultpath,args):
    # rankLearning -> output : result_faith
    dt = datetime.datetime.now()
    ## def of filename & filename2
    filename = str('{0:02d}'.format(dt.month)+'{0:02d}'.format(dt.day)+'_'+'{0:02d}'.format(dt.hour)+'{0:02d}'.format(dt.minute))
    if dt.minute == 59:
        filename2 = str('{0:02d}'.format(dt.month)+'{0:02d}'.format(dt.day)+'_'+'{0:02d}'.format(dt.hour+1)+'00')
    else:
        filename2 = str('{0:02d}'.format(dt.month)+'{0:02d}'.format(dt.day)+'_'+'{0:02d}'.format(dt.hour)+'{0:02d}'.format(dt.minute+1))
    ## newrankLearn.py
    os.system(
        'mpiexec -n '+str(args.n)+' python newrankLearn.py '+datapath+'/ranking.csv -e '+str(int(args.epoch/args.n))+' -rp '+str(args.rp)+' -rb '+str(args.rb)+' -d '+str(args.dim)+' -o '+str(faithresultpath)+' --mpi'
    )
    ## filename change -> './no_?/--brandids--/rprb_??_??'
    if os.path.exists(faithresultpath+'/'+filename):
        os.rename(faithresultpath+'/'+filename,faithresultpath+'/rprb_'+str(args.rp)+'_'+str(args.rb))
    elif os.path.exists(resultpath+'/'+filename2):
        os.rename(faithresultpath+'/'+filename2,faithresultpath+'/rprb_'+str(args.rp)+'_'+str(args.rb))
## evaluate
def faithfulness(playerpoint,resultpoint,faithno,args):
    # making prank.csv
    faithdistance = cdist(playerpoint,resultpoint)
    prankfile = args.path+'/result_faith/dim_'+str(args.dim)+'/no_'+str(faithno)+'/rprb_'+str(args.rp)+'_'+str(args.rb)+'/prank.csv'
    if os.path.isfile(prankfile):
        prank = csvopen(prankfile,'int')
    else:
        prank = np.empty((numplayers,numbrands),dtype='int64')
        print('prank matrix making...') # [TEST!]
        for i in tqdm(range(numplayers)):
            sort = [ np.argsort(faithdistance[i,:])[j] for j in range(numbrands) ]
            prank[i,:] = np.array(sort)
        ## save
        csvsave(prank,prankfile,'%d')
    # compare truth & pmatrix
    compare = np.array([0 for i in range(numbrands)])
    for i in range(numplayers):
        ithtruth = truthfullmatrix[i,:]
        ithplayer = prank[i,:]
        for j in range(numbrands):
            if ithtruth[j] == ithplayer[j]:
                compare[j] += 1
            else:
                break
    # save
    csvsave(compare,args.path+'/result_faith/dim_'+str(args.dim)+'/no_'+str(faithno)+'/rprb_'+str(args.rp)+'_'+str(args.rb)+'/compare.csv','%d')
    return compare



##################################################################################################
if __name__ == '__main__':
    start = time.time() # [TEST!]

    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help='directory path', default='./0710_sushi')
    parser.add_argument('-sb', '--selectbrands', nargs='+', type=int, help='学習させるbrandid (list)')
    parser.add_argument('-pn', '--expno', type=int, help='実験id', default=1)
    parser.add_argument('-n', type=int, help='並列数', default=4)
    parser.add_argument('-e', '--epoch', type=int, help='実験epoch数', default=200)
    parser.add_argument('-rp', type=float, help='-rp値', default=1e-09)
    parser.add_argument('-rb', type=float, help='-rb値', default=0.01)
    parser.add_argument('-d', '--dim', type=int, help='射影次元', default=2)
    parser.add_argument('-u', type=int, help='一様分布点の個数(領域の体積測定精度)', default=100000000)
    ## prediction?
    parser.add_argument('-enb', '--evaluatenumbrands', nargs='+', type=int, help='評価するbrand数 (list)')
    ## 各playerの上位Nbrandsのみを抽出(入力がなければbrand数そのものの順列で領域決定)
    parser.add_argument('-topn', '--topnumbrands', type=int, help='各player&samplepointの上位Nbrandsのみを抽出')
    ## 個々のfaithfulness? (8/19)
    parser.add_argument('-faith', type=int, help='第faith位までの正解率が何%かを調べる')
    parser.add_argument('-faithpn', type=int, help='faithfulness実験回数', default=5)
    args = parser.parse_args()

    # fullranking data set
    ## name & path
    csvdataname = str(''.join(map(str,args.selectbrands)))
    datapath = args.path+'/data/csv/'+csvdataname
    numbrands = len(args.selectbrands)
    renamebrandslist = list(range(numbrands))
    ## do
    if os.path.isfile(datapath+'/fullranking.csv'):
        truthfullmatrix = csvopen(datapath+'/fullranking.csv','int')
    else:
        os.mkdir(datapath)
        datamatrix = datamatrix(args)
        truthfullmatrix = select(datamatrix,args)
        csvsave(truthfullmatrix,datapath+'/fullranking.csv','%d')
    ## appendix
    numplayers = truthfullmatrix.shape[0]
    
    # ranking data set
    if os.path.isfile(datapath+'/ranking.csv'):
        rankmatrix = csvopen(datapath+'/ranking.csv','int')
    else:
        rankmatrix = rankingmatrix(truthfullmatrix)
        csvsave(rankmatrix,datapath+'/ranking.csv','%d')
    
    # rankLearning
    ## ready
    resultpath = args.path+'/result/no_'+str(args.expno)+'/'+str(csvdataname)+'/dim_'+str(args.dim)
    ## do
    if os.path.isfile(resultpath+'/rprb_'+str(args.rp)+'_'+str(args.rb)+'/out_brands0000.csv'):
        resultpoint = csvopen(resultpath+'/rprb_'+str(args.rp)+'_'+str(args.rb)+'/out_brands0000.csv','float')
    else:
        rankLearning(datapath,args)
        resultpoint = csvopen(resultpath+'/rprb_'+str(args.rp)+'_'+str(args.rb)+'/out_brands0000.csv','float')
    
    # evaluate
    if args.faith is None:
        if args.topnumbrands is None:
            ## set
            if args.evaluatenumbrands is None:
                checknblist = [int(numbrands)]
            else:
                checknblist = args.evaluatenumbrands
            ## do
            for checknb in checknblist:
                sushi = evaluate(resultpoint,truthfullmatrix,args,checknb)
                cormatrix = sushi[2]
        else:
            ## do
            topsushi = topevaluate(resultpoint,truthfullmatrix,args)
            topcormatrix = topsushi[2]
    else:
        # faithfulness
        ## prepare
        faithful = np.empty((args.faithpn,numbrands),dtype='int64')
        ## rankLearning -> analyze
        for faithno in range(args.faithpn):
            ## playerpoint
            ### ready
            faithresultpath = args.path+'/result_faith/dim_'+str(args.dim)+'/no_'+str(faithno)
            ### do
            if os.path.isfile(faithresultpath+'/rprb_'+str(args.rp)+'_'+str(args.rb)+'/out_players0000.csv'):
                playerpoint = csvopen(faithresultpath+'/rprb_'+str(args.rp)+'_'+str(args.rb)+'/out_players0000.csv','float')
            else:
                faithrankLearning(datapath,faithresultpath,args)
                playerpoint = csvopen(faithresultpath+'/rprb_'+str(args.rp)+'_'+str(args.rb)+'/out_players0000.csv','float')
            ## playerpoint -> distance -> prank -> compare
            if os.path.isfile(faithresultpath+'/rprb_'+str(args.rp)+'_'+str(args.rb)+'/compare.csv'):
                compare = csvopen(faithresultpath+'/rprb_'+str(args.rp)+'_'+str(args.rb)+'/compare.csv','float')
            else:
                compare = faithfulness(playerpoint,resultpoint,faithno,args)
            ## data copy
            faithful[faithno,:] = compare
        ## summary
        mean_value = faithful.mean(axis=0) # 平均値
        max_value = faithful.max(axis=0) # 最大値
        min_value = faithful.min(axis=0) # 最小値
        var_value = faithful.var(axis=0) # 分散値
        ## save
        summary = np.stack([mean_value,max_value,min_value,var_value])
        csvsave(summary,args.path+'/result_faith/dim_'+str(args.dim)+'/summary.csv','%f')
        ## plot
        ####
        ####
    # test
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

"""
<How to>

e.g.
(base) AkiranoMacBook-Pro:rankLearning-master akira$ python sushi.py -sb 0 1 2 3 -enb 3 -u 100 -d 3

## faithfulness(08/20 21:13記入)
python sushi.py -sb 0 1 2 3 4 5 6 7 8 9 -faith 10 -faithpn 3 -d 9

"""

