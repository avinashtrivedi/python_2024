from hmnet import ConvTransNet, BetterConvNet, SimpleConvNet, TransformerNoConv
from data_reader import HisModSingleDataset, HisModDataset

import torch
import torch.nn as nn

import scipy
from sklearn import metrics

import numpy as np

import matplotlib.pyplot as plt

import os, sys

import math

TRIAL_NAME = "betterConvTest"
TRIAL_DIR = "./checkpoints/" + TRIAL_NAME

print("====== GPU Info ======")

print("cuda available:", torch.cuda.is_available())

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE)

print("======================")

MAIN_DIR = "./ProcessedData/"

def train(
	hmnet, 
	train_loader, 
	val_loader, 
	checkpoint_name = "", 
	epoch = 10
	):

	optim = torch.optim.Adam(hmnet.parameters(), amsgrad=True, lr = 0.0003)

	checkpoint = "./checkpoints/" + checkpoint_name

	min_val_mse = float('inf')

	torch.set_grad_enabled(True)
	hmnet.train()

	print("train...")
	c = 0
	for i in range(epoch):
		for x, y in train_loader:

			x = x.to(DEVICE)
			y = y.to(DEVICE)

			input_mat = x.float()
			pred = hmnet(input_mat)
			pred = pred.squeeze()

			loss = nn.MSELoss()

			y = y.float()
			y = y.squeeze()

			out = loss(pred, y)

			out.backward()

			optim.step()
			optim.zero_grad()

		mse, r2, _ = eval(val_loader, hmnet)
		print("epoch:", (i+1))
		print("val pcc:", r2)

		if(mse < min_val_mse):
			min_val_mse = mse
			torch.save(hmnet.state_dict(), checkpoint)

		torch.set_grad_enabled(True)
		hmnet.train()	

	print("finished training")

	hmnet.load_state_dict(torch.load(checkpoint))

	return hmnet

#Eval for validation set across all cells
def eval(test_data, model):
	torch.set_grad_enabled(False)
	model.eval()
	pred_list = []; label_list = []
	for x, y in test_data:

		x = x.to(DEVICE)
		y = y.to(DEVICE)

		input_mat = x.float()

		pred = model(input_mat)
		y = y.float()

		pred_list.append(pred.squeeze().item())
		label_list.append(y.squeeze().item())

	pred_list = np.array(pred_list)
	label_list = np.array(label_list)

	#From https://github.com/QData/DeepDiffChrome
	R2,p=scipy.stats.pearsonr(label_list, pred_list)
	MSE=metrics.mean_squared_error(label_list, pred_list)
	return MSE, R2, p

#From https://github.com/QData/DeepDiffChrome/blob/master/data.py
def getlabel(c1,c2):
	# get log fold change of expression

	label1=math.log((float(c1)+1.0),2)
	label2=math.log((float(c2)+1.0),2)
	label=[]
	label.append(label1)
	label.append(label2)

	fold_change=(float(c2)+1.0)/(float(c1)+1.0)
	log_fold_change=math.log((fold_change),2)
	return (log_fold_change, label)

#Eval for test set for a cell pair
def test_eval(test_data, model):
	torch.set_grad_enabled(False)
	model.eval()
	pred_list = []; label_list = []
	for x1, x2, y in test_data:

		x1 = x1.to(DEVICE)
		x2 = x2.to(DEVICE)
		y = y.to(DEVICE)

		x1 = x1.float(); x2 = x2.float()

		#Get gene expression predictions
		pred1 = model(x1); pred2 = model(x2)

		y = y.float()

		pred1 = pred1.squeeze().item()
		pred2 = pred2.squeeze().item()

		#Get log fold change of gene expression predictions
		pred = getlabel(pred1, pred2)

		pred_list.append(pred[0])
		label_list.append(y.squeeze().item())

	pred_list = np.array(pred_list)
	label_list = np.array(label_list)

	#From https://github.com/QData/DeepDiffChrome
	R2,p=scipy.stats.pearsonr(label_list, pred_list)
	MSE=metrics.mean_squared_error(label_list, pred_list)
	return MSE, R2, p

def graph_results(pccs):
	labels = ["E123-E003", 
	"E116-E003", 
	"E123-E116", 
	"E003-E005", 
	"E003-E006",
	"E006-E007",
	"E005-E006",
	"E003-E004",
	"E004-E006",
	"E037-E038"]

	ticks = np.arange(10) + 1

	plt.figure()
	plt.xticks(ticks, labels, rotation = 45)
	plt.plot(ticks, pccs, linestyle='dashed', marker='o')
	plt.ylabel("Pearson Correlation Coefficient")
	plt.xlabel("Cell Pairs")
	plt.ylim(0, 1)
	plt.show()


if __name__ == '__main__':
	
	cells = [
	"E123",
	"E116", 
	"E003", 
	"E004",
	"E005", 
	"E006", 
	"E037", 
	"E038", 
	"E007"
	]

	#graph_results(simple_conv_pccs)

	if(os.path.exists(TRIAL_DIR)):
		print("Error: trial already exists. Please choose a different name.")
		sys.exit()
	else:
		os.makedirs(TRIAL_DIR)


	hmnet = BetterConvNet(use_abs = True)
	hmnet = hmnet.to(DEVICE)

	TRIAL_SAVE = TRIAL_NAME + "/" + TRIAL_NAME + "_SINGLE"

	train_datasets = []
	val_datasets = []
	test_datasets = []

	print("constructing dataset...")
	for cell in cells:	
		cellA_expr_file = cell + ".expr.csv"
		cellA_file = cell + ".train.csv"

		cellA_val = cell + ".valid.csv"

		print("loading data for:", cell)

		train_data = HisModSingleDataset(
			cellA_file, 
			cellA_expr_file, 
			MAIN_DIR
		)
		
		val_data = HisModSingleDataset(
			cellA_val, 
			cellA_expr_file, 
			MAIN_DIR
		)

		cellA_expr_file = cell + ".expr.csv"
		cellA_file = cell + ".test.csv"

		train_datasets.append(train_data)
		val_datasets.append(val_data)


	#Chain all cell data together for train + val set
	train_dataset = torch.utils.data.ConcatDataset(train_datasets)
	val_dataset = torch.utils.data.ConcatDataset(val_datasets)

	train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True)
	val_loader = torch.utils.data.DataLoader(val_dataset)

	print("data loaded!")

	hmnet = train(hmnet, train_loader, val_loader = val_loader, checkpoint_name = TRIAL_SAVE)

	cell_pairs = [
	["E123", "E003"],
	["E116", "E003"],
	["E123", "E116"],
	["E003", "E005"],
	["E003", "E006"],
	["E006", "E007"],
	["E005", "E006"],
	["E003", "E004"],
	["E004", "E006"],
	["E037", "E038"]
	]

	#Evaluate test set performancce for each cell pair
	for cell_pair in cell_pairs:
		cellA_expr_file = cell_pair[0] + ".expr.csv"
		cellA_file = cell_pair[0] + ".test.csv"
		cellB_file = cell_pair[1] + ".test.csv"
		cellB_expr_file = cell_pair[1] + ".expr.csv"

		test_dataset = HisModDataset(
			cellA_file, 
			cellA_expr_file, 
			cellB_file, 
			cellB_expr_file, 
			MAIN_DIR, 
			ignore_B = False
		)

		test_loader = torch.utils.data.DataLoader(test_dataset)

		_, R2, p = test_eval(test_loader, hmnet)
		print("eval on test set for cell pair" + str(cell_pair) + ":", R2, p)

	
