import numpy as np 
import pandas as pd 
import sys
from sklearn import cluster, datasets, metrics
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from scoring import getScore
from consensusClustering import *

def kmeans(data,k,seed):
	'''
	K means clustering

	Args:
		data (pd.DataFrame) : column for proteins(feature), raw for patients(sample)
		k (Int) : number of cluster
		seed (Int) : initial seed
	Return:
		cluster_labels (list) : cluster number of each sample
		silhouette_avg (float) : cluster silhouette score
	'''

	patient = data.index
	print(patient)
	patient_list = []

	for i in patient:
		p = list(data.iloc[i,1:])
		patient_list.append(p)
	kmeans_fit = cluster.KMeans(n_clusters = k,random_state=seed).fit(patient_list)
	cluster_labels = kmeans_fit.labels_
	silhouette_avg = metrics.silhouette_score(patient_list, cluster_labels)
	return cluster_labels, silhouette_avg

def hclust(data,k):
	'''
	Agglomerative Hierarchical clustering

	Args:
		data (pd.DataFrame) : column for proteins(feature), raw for patients(sample)
		k (Int) : number of cluster
	Return:
		cluster_labels (list) : cluster number of each sample
	'''
	patient = data.index
	patient_list = []

	for i in patient:
		p = list(data.iloc[i,1:])
		patient_list.append(p)
	clustering = AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto',
                        connectivity=None, distance_threshold=None,
                        linkage='ward', memory=None, n_clusters=k,
                        pooling_func='deprecated').fit(patient_list)
	return clustering.labels_

if __name__ == '__main__':

	data_path = sys.argv[1]
	LABELNAME = 'Stage'
	LABEL = pd.read_csv('./data/input/label.csv')
	data = pd.read_csv(data_path)
	print("file : {}\n".format(data_path))

	print("..... Kmeans Clustering outcome .....\n")
	cluster_label,score = kmeans(data,5,1)
	ari = getScore('ARI',cluster_label,labelname=LABELNAME,label=LABEL)
	# ami = getScore('AMI',cluster_result,labelname=LABELNAME,label=LABEL)
	mcc = getScore('MCC',cluster_label,labelname=LABELNAME,label=LABEL)
	gini = getScore('gini',cluster_label)
	print(cluster_label)
	print("ARI : {}\tMCC : {}\tGini : {}\t\n".format(ari,mcc,gini))

	print("..... Agglomerative Clustering outcome .....\n")
	cluster_label = hclust(data,5)
	ari = getScore('ARI',cluster_label,labelname=LABELNAME,label=LABEL)
	# # ami = getScore('AMI',cluster_result,labelname=LABELNAME,label=LABEL)
	mcc = getScore('MCC',cluster_label,labelname=LABELNAME,label=LABEL)
	gini = getScore('gini',cluster_label)
	print(cluster_label)
	print("ARI : {}\tMCC : {}\tGini : {}\t\n".format(ari,mcc,gini))

	print("..... Kmeans Consensus Clustering outcome .....\n")
	cc_data = data.iloc[:,1:].values.tolist()
	print(np.array(cc_data).shape)
	CC = ConsensusCluster(cluster=cluster.KMeans,L=3,K=9,H=10,resample_proportion=0.8).fit(data=np.array(cc_data))
	print('cluster number select by ConsensusClustering : {}'.format(CC.bestK))
	print(CC.deltaK)
	CC_predict = CC.predict()
	ari = getScore('ARI',CC_predict,labelname=LABELNAME,label=LABEL)
	mcc = getScore('MCC',CC_predict,labelname=LABELNAME,label=LABEL)
	gini = getScore('gini',CC_predict)
	print('predict cluster according to consensus matrix : ')
	print(CC_predict)
	print("ARI : {}\tMCC : {}\tGini : {}\t\n".format(ari,mcc,gini))

	CC_predict_data = CC.predict_data(data=np.array(cc_data))
	ari = getScore('ARI',CC_predict_data,labelname=LABELNAME,label=LABEL)
	mcc = getScore('MCC',CC_predict_data,labelname=LABELNAME,label=LABEL)
	gini = getScore('gini',CC_predict_data)
	print('predict cluster according to data : ')
	print(CC_predict_data)
	print("ARI : {}\tMCC : {}\tGini : {}\t\n".format(ari,mcc,gini))

	print("..... Agglomerative Consensus Clustering outcome .....\n")
	CC2 = ConsensusCluster(cluster=AgglomerativeClustering,L=3,K=9,H=10,resample_proportion=0.8).fit(data=np.array(cc_data))
	print('cluster number select by ConsensusClustering : {}'.format(CC2.bestK))
	print(CC2.deltaK)
	CC2_predict = CC2.predict()
	ari = getScore('ARI',CC2_predict,labelname=LABELNAME,label=LABEL)
	mcc = getScore('MCC',CC2_predict,labelname=LABELNAME,label=LABEL)
	gini = getScore('gini',CC2_predict)
	print('predict cluster according to consensus matrix : ')
	print(CC2_predict)
	print("ARI : {}\tMCC : {}\tGini : {}\t\n".format(ari,mcc,gini))

	CC2_predict_data = CC2.predict_data(data=np.array(cc_data))
	ari = getScore('ARI',CC2_predict_data,labelname=LABELNAME,label=LABEL)
	mcc = getScore('MCC',CC2_predict_data,labelname=LABELNAME,label=LABEL)
	gini = getScore('gini',CC2_predict_data)
	print('predict cluster according to data : ')
	print(CC2_predict_data)
	print("ARI : {}\tMCC : {}\tGini : {}\t\n".format(ari,mcc,gini))
	
	

	




