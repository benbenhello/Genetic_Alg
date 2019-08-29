import random 
import numpy as np
import pandas as pd
import time
import sys , getopt
import math

from transform import transform
from clustering import kmeans, hclust
from scoring import getScore
from multiprocessing import Pool





GENES = '''01'''
RAW_DATA = ''
LABEL = ''
LABELNAME = ''
POPULATION_SIZE = 10
CONVERGENCE = 0.001
GENERATION_SIZE = 10
INITIAL_FEATURE_NUM = 1000
K = 5

FEATURE_NUM = 0

class Individual(object): 
	''' 
	Class representing individual in population 
	'''
	def __init__(self, chromosome): 
		self.chromosome = chromosome 
		self.fitness, self.ari, self.ami, self.mcc, self.gini = self.cal_fitness() 

	@classmethod
	def mutated_genes(self): 
		''' 
		create random genes for mutation 
		'''
		global GENES 
		gene = random.choice(GENES) 
		return gene 

	@classmethod
	def create_gnome(self): 
		''' 
		create chromosome or string of genes 
		'''
		g = np.array(['1' for _ in range(INITIAL_FEATURE_NUM)]+['0' for _ in range(FEATURE_NUM - INITIAL_FEATURE_NUM)])
		np.random.shuffle(g)
		genome = list(g)
		# print(genome)
		return genome

	def mate(self, par2): 
		''' 
		Perform mating and produce new offspring 
		'''

		# chromosome for offspring 
		child_chromosome = [] 
		for gp1, gp2 in zip(self.chromosome, par2.chromosome):	
			 
			# random probability 
			prob = random.random() 

			# if prob is less than 0.45, insert gene 
			# from parent 1 
			if prob < 0.45: 
				child_chromosome.append(gp1) 

			# if prob is between 0.45 and 0.90, insert 
			# gene from parent 2 
			elif prob < 0.90: 
				child_chromosome.append(gp2) 

			# otherwise insert random gene(mutate), 
			# for maintaining diversity 
			else: 
				child_chromosome.append(self.mutated_genes()) 


		# create new Individual(offspring) using 
		# generated chromosome for offspring 
		return Individual(child_chromosome) 

	def cal_fitness(self): 
		''' 
		Calculate fittness score, 
		'''
		
		cluster_result, score = kmeans(transform(self.chromosome,RAW_DATA), K,1) 
		cluster_result = hclust(transform(self.chromosome,RAW_DATA), K)
		
		ari = getScore('ARI',cluster_result,labelname=LABELNAME,label=LABEL)
		# ami = getScore('AMI',cluster_result,labelname=LABELNAME,label=LABEL)
		ami = 0
		mcc = getScore('MCC',cluster_result,labelname=LABELNAME,label=LABEL)
		gini = getScore('gini',cluster_result)
		
		fitness = ari
		return fitness, ari, ami, mcc, gini

def help_message():
	print('\nUsage : python ga.py -i <raw_datafile_path> -l <labelfile_path> -n <labelname> -o <outputfile_path> [optional arguments]\n'\
			      '         -i <raw_datafile_path>      (.csv): Raws for samples, columns for features ; NOTE! first column should be sample name \n'
			      '         -l <labelfile_path>         (.csv): Raws for smaples, columns for Labels ; NOTE! first column should be sample name \n'\
			      '         -n <labelname>           : target label for calculate MCC score ; NOTE! must be same as label.csv column name\n'\
			      '         -o <outputfile_path>    (.csv): output feature subset of raw data DataFrame ;\n'\
			      '         -p <population_szie>     : default is 10\n'\
			      '         -g <generation_size>     : default is 10\n'\
			      '         -c <convergence_ratio>   : default is 0.001\n'\
			      '         -f <feature_num>         : default is 1000\n'\
			      '        [-h] print this message.')
	print('\n\n**** This program is Feature Selection Implement by Genetic Algorithm'\
				'\n       Main program is developed by ZeMing Chen '\
				'\n       fitness function is developed by ZiYi Dai ****\n')


def main(argv): 
	global RAW_DATA
	global LABEL
	global LABELNAME
	global POPULATION_SIZE 
	global GENERATION_SIZE
	global CONVERGENCE
	global INITIAL_FEATURE_NUM
	global FEATURE_NUM

	raw_datafile = ''
	labelfile = ''
	labelname = ''
	outputfile = ''

	try:
		opts, args = getopt.getopt(argv,"hi:l:n:o:p:g:c:f:",["ifile=","lfile=","lname=","ofile=","psize=","gsize=","convergence=","fnum="])
	except getopt.GetoptError:
		help_message()
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			help_message()
			sys.exit()

		elif opt in ("-i", "--ifile"):
			raw_datafile = str(arg)
		elif opt in ("-l", "--lfile"):
			labelfile = str(arg)
		elif opt in ("-n", "--lname"):
			labelname = str(arg)
		elif opt in ("-o", "--ofile"):
			outputfile = str(arg)
		elif opt in ("-p", "--psize"):
			POPULATION_SIZE = int(arg)
		elif opt in ("-g", "--gsize"):
			GENERATION_SIZE = int(arg)
		elif opt in ("-c", "--convergence"):
			CONVERGENCE = float(arg)
		elif opt in ("-f", "--fnum"):
			INITIAL_FEATURE_NUM = int(arg)
		
		
	if raw_datafile == '' or labelfile == '' or labelname == '' or outputfile == '':
		help_message()
		sys.exit()
	
	RAW_DATA = pd.read_csv(raw_datafile)
	LABEL = pd.read_csv(labelfile)
	LABELNAME = labelname
	FEATURE_NUM = len(RAW_DATA.columns)-1
	
	#initial generation setting 
	generation = 1
	generation_list = []

	# create initial population
	# use multiprocessing to speed up
	p =  Pool(processes=4)
	a = time.time()

	population = []
	for _ in range(POPULATION_SIZE): 
		r = p.apply_async(Individual,(''.join(Individual.create_gnome()),))
		population.append(r)
	p.close()
	p.join()

	b = time.time()
	
	print("initial population create time :{}\t\n".format(b-a))
	
	
	while True:
		# sort the population in decreasing order of fitness score 
		
		population = sorted(population, key = lambda x:x.get().fitness, reverse=True) 
		print("Generation: {}\tString: {}\tFitness: {}\tARI: {}\tAMI: {}\tMCC: {}\tgini: {}\n".format(generation,"".join(population[0].get().chromosome[:10]),
																	population[0].get().fitness,
																	population[0].get().ari,
																	population[0].get().ami,
																	population[0].get().mcc,
																	population[0].get().gini)) 
		generation_list.append(population[0].get().fitness)		
		if (generation%GENERATION_SIZE) == 0:
			if (generation_list[generation-1] - generation_list[generation-GENERATION_SIZE ]) < CONVERGENCE:
				print("terminate")
				print("improvement between generation {} & generation {} is {} ,less than convergence ratio {}".format(generation,(generation-GENERATION_SIZE+1 ),
																	(generation_list[generation-1] - generation_list[generation-GENERATION_SIZE ]),CONVERGENCE))
				break
		
		#generate new offsprings for new generation 
		new_generation = [] 

		# Perform Elitism, that mean 10% of fittest population 
		# goes to the next generation 
		s = int((10*POPULATION_SIZE)/100) 
		new_generation.extend(population[:s]) 
		s = int((70*POPULATION_SIZE)/100) 

		# From 50% of fittest population, Individuals 
		# will mate to produce offspring 
		a = time.time()
		p2 = Pool(processes=8)
		for _ in range(s): 
			
			r2 = p2.apply_async(random.choice(population[:int(POPULATION_SIZE/2)]).get().mate,(random.choice(population[:int(POPULATION_SIZE/2)]).get(),))
			new_generation.append(r2)

		s = int((20*POPULATION_SIZE)/100)
		for _ in range(s):
			r = p2.apply_async(Individual,(''.join(Individual.create_gnome()),))
			new_generation.append(r)
		
		population = new_generation
		p2.close()
		p2.join()
		b = time.time()
		 
		print("New population create time :{}\t".format(b-a))

		generation += 1

		

	output = transform(population[0].get().chromosome, RAW_DATA)
	output.to_csv(str(outputfile),index =False) 
	

if __name__ == '__main__': 
	main(sys.argv[1:]) 

