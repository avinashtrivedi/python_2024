import torch
import torch.nn as nn
import torch.utils.data

import numpy as np
import pandas as pd

import math

MAIN_DIR = "./ProcessedData/"

class HisModDataset(torch.utils.data.Dataset):
	def __init__(
		self, 
		cellA_file, 
		cellA_expr_file, 
		cellB_file, 
		cellB_expr_file, 
		main_dir, 
		ignore_B=False,
		shuffle_cols = False #experimental
		):
        
        print('in constructor')

		cell_cols = ["A", "B", "C", "D", "E", "F"] #dummy cols
		expr_cols = ["A", "B"] #dummy cols 

		#read in data into dataframes
		cellA_df = pd.read_csv(main_dir + cellA_file, names=cell_cols)
		cellB_df = pd.read_csv(main_dir + cellB_file, names=cell_cols)
		cellA_expr_df = pd.read_csv(main_dir + cellA_expr_file, names=expr_cols)
		cellB_expr_df = pd.read_csv(main_dir + cellB_expr_file, names=expr_cols)

		#200 entries per gene
		self.offset = 200
		self.length = len(cellA_df)//self.offset
		
		#all cols except one with gene info
		hm_cols = ["B", "C", "D", "E", "F"]

		#convert to tensor
		self.cellA_tensor = torch.tensor(cellA_df[hm_cols].values)
		self.cellB_tensor = torch.tensor(cellB_df[hm_cols].values)

		#using expr file, maps gene id -> expr value
		self.gene_to_valA = dict(zip(cellA_expr_df.A, cellA_expr_df.B))
		self.gene_to_valB = dict(zip(cellB_expr_df.A, cellB_expr_df.B))

		#extract gene info col from the train/test/val data
		self.geneA_names = cellA_df["A"]
		self.geneB_names = cellB_df["A"]

		self.ignore_B = ignore_B
		self.shuffle_cols = shuffle_cols

	def __getitem__(self, idx):
        print('in get items')
		idx = idx*self.offset

		#find the relevant slice of the tensor based on the id
		tensorA = self.cellA_tensor[idx:idx+self.offset]
		tensorB = self.cellB_tensor[idx:idx+self.offset]

		#transpose b/c data is initially 200x5
		tensorA = torch.transpose(tensorA, 0, 1)
		tensorB = torch.transpose(tensorB, 0, 1)

		if(self.shuffle_cols): #experimental
			tensorA = tensorA[:, torch.randperm(tensorA.size()[1])]
			tensorB = tensorB[:, torch.randperm(tensorB.size()[1])]

		#just takes the first gene window id corresponding to the 
		#group of 200 and uses it to find the gene id
		geneA = self.geneA_names[idx].split("_")[0]
		geneB = self.geneB_names[idx].split("_")[0]

		#print(geneA); print(geneB) #<-- should be the same

		#Get the expression values for cell A and cell B
		cA = self.gene_to_valA[geneA]
		cB = self.gene_to_valB[geneB]

		if(self.ignore_B):
			return tensorA, tensorB, cA

		#Find the log-fold change
		label = self.getlabel(cA, cB) 

		return tensorA, tensorB, label[0]

	def __len__(self):
		return self.length

	#From https://github.com/QData/DeepDiffChrome/blob/master/data.py
	def getlabel(self, c1,c2):
        print('in get label')
		# get log fold change of expression

		label1=math.log((float(c1)+1.0),2)
		label2=math.log((float(c2)+1.0),2)
		label=[]
		label.append(label1)
		label.append(label2)

		fold_change=(float(c2)+1.0)/(float(c1)+1.0)
		log_fold_change=math.log((fold_change),2)
		return (log_fold_change, label)

class HisModSingleDataset(torch.utils.data.Dataset):
	def __init__(
		self, 
		cellA_file, 
		cellA_expr_file, 
		main_dir
		):

		cell_cols = ["A", "B", "C", "D", "E", "F"] #dummy cols
		expr_cols = ["A", "B"] #dummy cols 

		#read in data into dataframes
		cellA_df = pd.read_csv(main_dir + cellA_file, names=cell_cols)
		cellA_expr_df = pd.read_csv(main_dir + cellA_expr_file, names=expr_cols)

		#200 entries per gene
		self.offset = 200
		self.length = len(cellA_df)//self.offset
		
		#all cols except one with gene info
		hm_cols = ["B", "C", "D", "E", "F"]

		#convert to tensor
		self.cellA_tensor = torch.tensor(cellA_df[hm_cols].values)

		#using expr file, maps gene id -> expr value
		self.gene_to_valA = dict(zip(cellA_expr_df.A, cellA_expr_df.B))

		#extract gene info col from the train/test/val data
		self.geneA_names = cellA_df["A"]

	def __getitem__(self, idx):

		idx = idx*self.offset

		#find the relevant slice of the tensor based on the id
		tensorA = self.cellA_tensor[idx:idx+self.offset]

		#transpose b/c data is initially 200x5
		tensorA = torch.transpose(tensorA, 0, 1)

		#just takes the first gene window id corresponding to the 
		#group of 200 and uses it to find the gene id
		geneA = self.geneA_names[idx].split("_")[0]

		#print(geneA); print(geneB) #<-- should be the same

		#Get the expression values for cell A and cell B
		cA = self.gene_to_valA[geneA]

		return tensorA, cA

	def __len__(self):
		return self.length

if __name__ == '__main__':
	cellA_expr_file = "E003.expr.csv"
	cellA_file = "E003.train.csv"
	cellB_file = "E004.train.csv"
	cellB_expr_file = "E004.expr.csv"

	print("loading data...")
	dataset = HisModDataset(cellA_file, cellA_expr_file, cellB_file, cellB_expr_file, MAIN_DIR)

	dataloader = torch.utils.data.DataLoader(dataset)
	i = 0
	for xA, xB, y in dataloader:
		i+=1

	print(i)
	print("data loaded!")
