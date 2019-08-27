import pandas as pd
import numpy as np 
import random
import sys , getopt
from operator import add
from sklearn import cluster, datasets, metrics
import matplotlib.pyplot as plt
from scoring import getScore
from clustering import kmeans, hclust

def barplot(label,data,output,t):
	N = 5
	labels = ['1','2','3','4','5']
	patient = data.index
	target = np.array(label)
	uni_target = np.unique(target)
	target_dic = {}
	for i in range(len(uni_target)):
		target_dic.update({uni_target[i]:i})
	bar_value = [[0 for j in range(N)] for i in range(len(uni_target))]
	for i in range(len(patient)):
		bar_value[(target_dic[target[i]])][(data.iloc[i,-1])]+=1
		
	# print(bar_value)

	ind = np.arange(N)    # the x locations for the groups
	width = 0.35       # the width of the bars: can also be len(x) sequence

	fig, ax = plt.subplots()
	p1 = ax.bar(ind, bar_value[0], width,label='I')
	p2 = ax.bar(ind, bar_value[1], width,label='II',bottom=np.array(bar_value[0]))
	p3 = ax.bar(ind, bar_value[2], width,label='III',bottom=np.array(bar_value[0])+np.array(bar_value[1]))
	p4 = ax.bar(ind, bar_value[3], width,label='IV',bottom=np.array(bar_value[0])+np.array(bar_value[1])+np.array(bar_value[2]))

  # Add some text for labels, title and custom x-axis tick labels, etc.
	ax.set_ylabel('count')
	ax.set_title(t)
	ax.set_xticks(ind)
	ax.set_xticklabels(labels)
	ax.legend()


	def autolabel(rects,l):
		height_list = l
		for i in range(len(rects)):
			height = height_list[i]+(rects[i].get_height()/2)
			height_list[i] = height_list[i]+rects[i].get_height()
			if rects[i].get_height() == 0:
				ax.annotate('{}'.format(''),xy=(rects[i].get_x() + rects[i].get_width() / 2, height),
					xytext=(0, -5),textcoords="offset points",ha='center', va='bottom')
			else :
				ax.annotate('{}'.format(rects[i].get_height()),xy=(rects[i].get_x() + rects[i].get_width() / 2, height),
					xytext=(0, -5),textcoords="offset points",ha='center', va='bottom')
		return height_list

	p1_height = autolabel(p1,[0,0,0,0,0])
	p2_height = autolabel(p2,p1_height)
	p3_height = autolabel(p3,p2_height)
	p4_height = autolabel(p4,p3_height)
	plt.savefig(output)

if __name__ == '__main__':
	data = pd.read_csv('./data/output/test.csv')
	result = pd.read_csv('./data/input/label.csv')
	cluster_label = hclust(data,5)
	result['class'] = cluster_label
	barplot(result['Stage'],result,'./data/output/test.jpg','test')
