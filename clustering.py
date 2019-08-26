import numpy as np 
import pandas as pd 
from sklearn import cluster, datasets, metrics
from sklearn.cluster import AgglomerativeClustering

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
	data = pd.read_csv('./data/input/data.csv')
	print("..... Kmeans Clustering outcome .....\n")
	cluster_label,score = kmeans(data,5,1)
	print(cluster_label)
	print(score)
	print("\n..... Agglomerative Clustering outcome .....\n")
	cluster_label = hclust(data,5)
	print(cluster_label)