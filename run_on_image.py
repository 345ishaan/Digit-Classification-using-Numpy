import sys
import os
from nn import *
import cv2

def run_on_image(img_path,model_path):

	net = NN(mode='test',model_path=model_path)

	img = cv2.imread(img_path)

	img = img[:,:,0].astype(np.float32)/255.0 - 0.5
	img = img.reshape(1,784)

	cache = net.forward(img)
	probs = cache[-2]

	label = np.argmax(probs,axis=1)

	print "Predicted Label {}".format(label[0])


if __name__ == '__main__':

	img_path = sys.argv[1]
	model_path = sys.argv[2]
	run_on_image(img_path,model_path)
