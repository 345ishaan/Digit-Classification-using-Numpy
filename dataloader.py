import sys
import os
import csv
import numpy as np

def load_data(dpath,mode='train'):
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
	import cv2
	dpath = '/home/ishan/code/python/cerebras/data'
	data = load_data(dpath,'test')
	save_path = './images'
	for i in range(20):
		img = data[i,1:].astype(np.uint8).reshape(28,28,1)
		print img.shape
		cv2.imwrite(os.path.join(save_path,'img_{}.png'.format(i)),img)

