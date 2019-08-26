import pandas as pd 
import numpy as np 
from scipy.special import comb
from sklearn.metrics import *
import numpy.linalg as LA
from enum import Enum
import math
import scipy.spatial as spatial

"""Lists of all ScoreTypes available"""
ScoreType = {}
ScoreType['Ext'] = ['rand_index','RI','adjust_rand_index','ARI','normalized_mutual_info','adjusted_mutual_info','AMI','v_measure','FMI','jaccard','MCC','pair_f_measure','rms_distance']
ScoreType['Int'] = ['silhouette','partition_index','PI','separation_index','SI']
ScoreType['Multi-Ext'] = ['multi_rand_index','multi_RI','multi_FMI','TP_percent','TN_percent','multi_jaccard','multi_MCC']
ScoreType['Others'] = ['gini','variance']


def _getGT(label,labelname):
    """
    Convert label column from str to ints

    Args:
        label (pd.DataFrame): total DataFrame of the labels
        labelname (string): one of the column names in label

    Return:
        ground_truth (np.array):the labels for patients
        len(types): how many different classes in ground_truth
    """
    col = label[labelname].values
    ground_truth = []
    types = []
    for i in range(len(col)):
        for j in range(len(types)):
            if col[i] == types[j]:
                ground_truth.append(j)
        if len(ground_truth)!= i+1:
            ground_truth.append(len(types))
            types.append(col[i])

    return ground_truth, len(types)


def _gini_coefficient(preds):
    counts = np.unique(preds,return_counts=True)[1]
    gini = 0
    for i in range(len(counts)):
        for j in range(len(counts)):
            gini += np.abs(counts[i] - counts[j])
    gini /= (2*len(counts))
    gini /= np.sum(counts)
    return gini

def _variance(preds):
    counts = np.unique(preds,return_counts=True)[1]
    return np.var(counts)

def _getPairTypeNum(clusters,classes):
    """
    Calculate the TP, TN, FP, FN pairs 

    Args:
        clusters(np.array): the cluster results (hard assignment), dimension [data_num,]
        classes(np.array): the label for a single labelname, dimension [data_num,]
    """
    tp_plus_fp = comb(np.bincount(clusters), 2).sum()
    tp_plus_fn = comb(np.bincount(classes), 2).sum()
    A = np.c_[(clusters, classes)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(clusters))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return {'TP':tp,'FP':fp,'FN':fn,'TN':tn}

#https://stackoverflow.com/questions/49586742/rand-index-function-clustering-performance-evaluation
def _rand_index(clusters,classes):
    """Calculated rand index score of clusters and classes"""
    res = _getPairTypeNum(clusters,classes)
    return (res['TP'] + res['TN']) / (res['TP'] + res['FP'] + res['FN'] + res['TN'])

def _jaccard_index(clusters,classes):
    res = _getPairTypeNum(clusters,classes)
    return res['TP'] / (res['TP'] + res['FP'] + res['FN'])

def _pair_f_measure(clusters,classes):
    res = _getPairTypeNum(clusters,classes)
    return 2*res['TP'] / (2*res['TP'] + res['FP'] + res['FN'])


def _matthews_correlation_coefficient(clusters,classes):
    """
    Calculate the MCC score 

    Args:
        clusters(np.array): the cluster results (hard assignment), dimension [data_num,]
        classes(np.array): the label for a single labelname, dimension [data_num,]
    """
    #print('calling MCC...')
    res = _getPairTypeNum(clusters,classes)
    N = res['TP'] + res['TN'] + res['FP'] + res['FN']
    S = (res['TP'] + res['FN'])/N
    P = (res['TP'] + res['FP'])/N
    numerator = res['TP']/N - S*P
    dominator = math.sqrt(P*S*(1-S)*(1-P))
    return numerator/dominator

def _rms_distance(clusters,classes):
    """
    Return the rms distance between 2 partitions.
    The ratio between agreements(TP+TN) and disagreements(FP+FN).
    The ratio is normalized by the expected agreements and expected disagreements.
    
    Args:
        clusters(np.array): the cluster results (hard assignment), dimension [data_num,]
        classes(np.array): the label for a single labelname, dimension [data_num,]
    """
    res = _getPairTypeNum(clusters,classes)
    actual_agree = res['TP'] + res['TN']
    actual_disagree = res['FN'] + res['FP']
    total_pairs = comb(len(classes),2)
    classes_prob = np.divide(comb(np.bincount(classes),2), total_pairs ).sum()
    cluster_prob = np.divide(comb(np.bincount(clusters),2), total_pairs ).sum()
    expected_agree = total_pairs * classes_prob * cluster_prob
    expected_disagree = total_pairs * ( classes_prob*(1-cluster_prob) + cluster_prob*(1-classes_prob))  
    return (actual_disagree/expected_disagree) / (actual_agree/expected_agree)

def normFuzzPartScore(u):
    """Returns Normalized Fuzzy Partition Score"""
    fpc = np.square(u).sum()/u.shape[1]
    score = (fpc-1/u.shape[0])/(1-1/u.shape[0])
    return score

def _separationIndex(X,cntr,u,m=2):
    """
    Calculate the partition index of the clustering

    Args:
        X(np.array): data, dimension [data_num feature_num]
        cntr(np.array): cntr of the clusters, dimension [cluster_num, feature_num]
        u(np.array): weighted membership of the data to each cluster, dimension [cluster_num, data_num]
        
    Return:
        return partition index
    """
    data_num = X.shape[0]       #(data_num, feature_num)
    cluster_num = cntr.shape[0] #(cluster_num, feature_num)
    #u: (cluster_num, data_num)

    compact = 0
    cntr_dist = []
    for i in range(cluster_num):
        for j in range(i+1,cluster_num):
            cntr_dist.append(LA.norm(cntr[i]-cntr[j]))
        
        w = np.power(u[i],m)
        comp_clus = 0
        for k in range(data_num):
            comp_clus += w[k] * LA.norm(X[k]-cntr[i])
        compact += comp_clus

    cntr_dist = np.array(cntr_dist)
    S =  (compact/(data_num*np.min(cntr_dist)))

    return S

def _partitionIndex(X,cntr,u,m=2):
    """
    Calculate the partition index of the clustering

    Args:
        X(np.array): data, dimension [data_num feature_num]
        cntr(np.array): cntr of the clusters, dimension [cluster_num, feature_num]
        u(np.array): weighted membership of the data to each cluster, dimension [cluster_num, data_num]
        
    Return:
        return partition index
    """
    data_num = X.shape[0] #(data_num, feature_num)
    cluster_num = cntr.shape[0] #(cluster_num, feature_num)
    SC = 0
    data_num_in_clus = u.sum(axis=1).reshape(-1)

    for i in range(cluster_num):
        cntr_dist = 0
        for j in range(cluster_num):
            cntr_dist += LA.norm(cntr[i]-cntr[j])

        w = np.power(u[i],m)
        compact = 0
        for k in range(data_num):
            compact += w[k] * LA.norm(X[k]-cntr[i])

        SC += (compact/(data_num_in_clus[i]*cntr_dist))

    return SC

def _hard_cluster_aug(preds,cluster_num):
    """
    Convert predict labels into probabilities, 
    i.e. convert matrix with 0 and 1 into a np.array, dimension [cluster_num data_num]
    
    Args:
        preds (np.array): Dimension [data_num, ], predicted labels for data, 
                        the cluster number starts from 0
        cluster_num (int): number of clusters in preds

    Return:
        u.T (np.array): u for fuzzy cluster results, dimension [cluster_num data_num]
    """
    u = np.zeros((preds.shape[0], cluster_num)) #u: (cluster_num, data_num)
    for i in range(preds.shape[0]):
        u[i][preds[i]] = 1
    #print(u.T.shape)

    return u.T

def _hard_cluster_aug_soft(X,center,metric):
    """
    Augment the membership value matrix from hard assignment to soft assignment
    The membership value is inversely proportional to distance (distance from the data point to each center) square.
    
    Args:
        X(np.array): data, dimension [data_num feature_num]
        center(np.array): center of the clusters, dimension [cluster_num, feature_num],
                        the ith entry of center is the center of the cluster with cluster num i  
        metric(string): the distance metric. P
    Return:
        u.T(np.array): u for fuzzy cluster results, dimension [cluster_num data_num]
    """
    n = center.shape[0]
    dist_names = {'manhattan':'cityblock', 'pearson':'correlation'}
    try:
        distance_matrix = spatial.distance.cdist(X,center,metric=metric)
    except:
        distance_matrix = spatial.distance.cdist(X,center,metric=dist_names[metric])
    
    distance_matrix = distance_matrix/(np.sum(distance_matrix,axis=1).reshape(-1,1))
   
    with np.errstate(divide='ignore', invalid='ignore'):
        u = 1/np.square(distance_matrix)
        u[distance_matrix == 0] = 1
    u = u/(np.sum(u,axis=1).reshape(-1,1)) #[data_num, cluster_num]


    return u.T

def _getCenter(X,preds,n):
    """
    Get the center of each cluster.

    Args:
        X(np.array): data values, dimension [data_num, feature_num]
        preds(np.array): dimension [data_num, ], the number in preds starts from 0,
                        no integer is skipped
        n(int): number of clusters

    Return:
        center(np.array): center of the clusters, dimension [cluster_num, feature_num],
                        the ith entry of center is the center of the cluster with cluster num i 
    """
    center = np.zeros((n,X.shape[1]))
    clus_mem_num = np.zeros(n)

    for i in range(len(preds)):
        center[preds[i]] += X[i]
        clus_mem_num[preds[i]] += 1

    center = np.divide(center.T,clus_mem_num).T
    return center

def _resetClusterNum(preds):
    """
    Make sure the cluster number in preds starts with 0 and no integer is skipped,
    i.e. np.unique(preds) = [0, 1, 2, 3...].

    Args: 
        preds(np.array): cluster results, dimension [data_num, ],
                        the cluster indices have to be continuous integers

    Return:
        adjusted(np.array): preds that follows the format mentioned above
        cluster_num(int): number of clusters in the prediction results
    """
    adjusted = preds - np.min(preds)
    assert len(np.unique(adjusted)) == (np.max(adjusted)+1)

    return adjusted, len(np.unique(adjusted))

def _getExtScore(score_type,preds,classes):
    """
    Return the score of a single label external validation.
    The cluster results must be hard assignments.

    Args:
        score_type(string): score type of the score
        preds(np.array): dimension [data_num], the cluster results
        classes(np.array): dimension [data_num], the label to be evaluated. 
                            The labels are already converted to integers.
    Return:
        score(float)
    """
    if score_type in ['rand_index','RI']:
        return _rand_index(preds, classes)
    elif score_type in ['adjust_rand_index','ARI']:
        return adjusted_rand_score(preds, classes)
    elif score_type == 'normalized_mutual_info':
        return normalized_mutual_info_score(preds,classes)
    elif score_type in ['adjusted_mutual_info','AMI']:
        return adjusted_mutual_info_score(preds,classes)
    elif score_type == 'v_measure':
        return homogeneity_completeness_v_measure(classes, preds)[2]
    elif score_type == 'FMI':
        return fowlkes_mallows_score(classes, preds)
    elif score_type == 'jaccard':
        return _jaccard_index(preds,classes)
    elif score_type == 'MCC':
        return _matthews_correlation_coefficient(preds,classes)
    elif score_type == 'pair_f_measure':
        return _pair_f_measure(preds,classes)
    elif score_type == 'rms_distance':
        return _rms_distance(preds,classes)

def _getExtScoreWeighted(score_type,preds,label,weights):
    """
    Calculate the weighted average score of the score type based on different labels

    Args:
        score_type(string): score type of the score
        preds(np.array): dimension [data_num], the cluster results
        label (pd.DataFrame): ground truths of the data, multilabels, dimension [data_num label_num]
        weights(list): weights for weighted average of the scores between different labels

    Return:
        score: the weighted average score of the score type
    """
    assert len(label.columns) == len(weights)
    scores = []
    for labelname in label.columns:
        classes, num_groups = _getGT(label, labelname=labelname)
        scores.append(_getExtScore(score_type,preds,classes))

    scores = np.array(scores)
    weights = np.array(weights)
    score = np.multiply(scores,weights).sum()/weights.sum()

    return score

def _getIntScore(score_type,preds,data,cluster_num,**kwargs):
    """
    Calculate the score for internal validation

    Args:
        score_type: string with the name of scoring function
        preds (np.array): cluster results of data, should not contain noise (-1)
        data (pd.DataFrame) : data of the patients, index are patient names,
                                columns are protein names
        kwargs:
            aug_type(string): which type of augmentation for the membership value matrix.
                            Only used for Partition Index and Separation Index.
                            Possible options: 'hard', 'soft'. Default is 'hard'
            distance_metric(string): which type of distance metric. Only required when aug_type is soft
    Return:
        score(float)
    """
    if score_type == 'silhouette':
            return silhouette_score(data.values, preds, metric='euclidean')

    elif score_type in ['partition_index','PI','separation_index','SI']:
        #internal validation with soft assignments required
        cntr = _getCenter(data.values,preds,cluster_num)

        aug_type = kwargs['aug_type']

        if aug_type == 'soft':
            metric = kwargs['distance_metric']
            u = _hard_cluster_aug_soft(data.values,cntr,metric=metric)

        elif aug_type == 'hard':
            u = _hard_cluster_aug(preds,cluster_num)
        else:
            print('No such aug type!!!')
        
        if score_type in ['partition_index','PI']:
            return _partitionIndex(data.values,cntr,u)
        else:
            return _separationIndex(data.values,cntr,u)

def getSimilarity(score_type,partitions):
    """
    Get the similarity between 2 cluster results.

    Args:
        score_type(string): possible options: rms_distance, MCC, ARI, AMI
        partitions(np.array): dimension [data_num, 2], the 2 columns are 2 different cluster results
                            The entries should be non-negative integers, each represents a cluster.
                            The integers should be continous.
    Return:
        score
    """
    part1, cluster_num1 = _resetClusterNum(partitions[:,0].reshape(-1))
    part2, cluster_num2 = _resetClusterNum(partitions[:,1].reshape(-1))

    if score_type == 'MCC':
        return _matthews_correlation_coefficient(part1,part2)
    elif score_type == 'ARI':
        return adjusted_rand_score(part1, part2)
    elif score_type == 'AMI':
        return adjusted_mutual_info_score(part1, part2)
    elif score_type == 'rms_distance':
        return _rms_distance(part1,part2)
    else:
        print('No such score type for similarity!!')

def getScore(score_type,preds,**kwargs):
    """
    Get score from different scoring functions.
    Since the cluster results are "hard assignments", if the score_type is partition_index or separation_index,
    the results will be converted into a matrix containing only 0 and 1.

    Args:
        score_type: string with the name of scoring function
        preds (np.array): cluster results of data, should not contain noise
        kwargs: parameters for different score type, can possibly contain
            data (pd.DataFrame) : data of the patients, index are patient names,
                                columns are protein names,
                                only used when the scoring function is an internal validation
            label (pd.DataFrame): ground truths of the data, multilabels, dimension [data_num label_num]
            labelname (string): string of the label name being evaluated.
                                Another possible option: weighted_  
            weights(list): weights for weighted average of the scores between different labels
            others:
                ex: k for 'combination' relation_type
                    weight_list, threshold for 'weighted' relation type
            aug_type(string): which type of augmentation for the membership value matrix.
                            Only used for Partition Index and Separation Index.
                            Possible options: 'hard', 'soft'. Default is 'hard'
            distance_metric(string): which type of distance metric. Only required when aug_type is soft
    Return:
        score of the clustering
    """
    preds, cluster_num = _resetClusterNum(preds)

    if score_type in ScoreType['Int']:
        data = kwargs.pop('data')
        return _getIntScore(score_type,preds,data,cluster_num,**kwargs)

    elif score_type in ScoreType['Ext']:
        label = kwargs['label']
        labelname = kwargs['labelname']
        if labelname != 'weighted_':
            #single label external validation
            classes, num_groups = _getGT(label, labelname=labelname)
            return _getExtScore(score_type,preds,classes)
        else:
            weights = kwargs['weights']
            return _getExtScoreWeighted(score_type,preds,label,weights)
            
    elif score_type in ScoreType['Others']:
        if score_type == 'gini':
            return _gini_coefficient(preds)
        elif score_type == 'variance':
            return _variance(preds)

    else:
        print("No such score type")

def getfuzzScore(score_type,u,cntr,X,label,labelname):
    """
    Get score from different scoring functions, cluster results are fuzzy

    Args:
        score_type: string with the name of scoring function
        u(np.array): dimension [cluster_num data_num], result from fuzzy clustering
        cntr(np.array): centers of the fuzzy clustering, dimension [cluster_num feature_num]
        X (np.array): data of the patients, dimension [data_num feature_num]
        label (pd.DataFrame): ground truths of the data, multilabels, dimension [data_num label_num]
        labelname (string): string of the label name being evaluated

    Return:
        score of the fuzzy clustering
    """
    if score_type == 'FU':
        return normFuzzPartScore(u)

    elif score_type == 'partition index':
        return _partitionIndex(X,cntr,u)

    elif score_type == 'separation index':
        return _separationIndex(X,cntr,u)

    else:
        print("No such score type")




if __name__=='__main__':
    from data import readfile
    data,label = readfile()
    print('data shape', data.shape)
    print('label shape',label.shape)
    print(data.iloc[:10,:2])

    X = data.drop(['Name'],axis=1).values
    print(X.shape)


