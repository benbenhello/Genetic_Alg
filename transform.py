import numpy as np 
import pandas as pd 

def transform(genome,raw_data):
	'''
	transform chromesome(binary list) to correspond dataframe which is clustering input

	Args:
		genome (str) : Individual from genetic alg
		raw_data (pd.DataFrame) : user input file
	
	Return:
		data (pd.DataFrame) : target feature subset of raw data, clustering input
	'''
	
	data = raw_data
	column = data.columns
	protein_list = column[1:]
	drop_list = []
	for i in range(len(genome)):
		if genome[i] == '0':
			drop_list.append(protein_list[i])
	data = data.drop(drop_list,axis=1)
	return data