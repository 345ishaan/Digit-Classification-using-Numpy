import sys
import os
import csv
import numpy as np

def load_data(dpath,mode='train'):
	'''
	Load the CSV data
	If mode == 'train' , Loads from mnist_train.csv
	If mode == 'test', Loads from mnist_test.csv
	'''
	if mode == 'train':
		file_name = os.path.join(dpath,'mnist_train.csv')
	else:
		file_name = os.path.join(dpath,'mnist_test.csv')
	data=[]
	
	try:
		fp = open(file_name,'rb')
		reader = csv.reader(fp)
	except:
		raise Exception("Invalid Data Path")

	print "Loading Dataset===>"
	
	for row in reader:
		data.append(row)
	print "Done!"
	data = np.asarray(data)
	return data

if __name__ == '__main__':
	dpath = sys.argv[1]
	mode = sys.argv[2]
	data = load_data(dpath,mode)
