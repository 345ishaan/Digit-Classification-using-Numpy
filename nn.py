import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from dataloader import *

class NN(object):

	def __init__(self,hidden_dims=(1024,2048),n_hidden=2,mode='train',dpath=None,model_path=None):
		
		
		if dpath is not None:
			try:
				self.mnist_data = load_data(dpath,mode=mode)
				self.n_samples,self.ip_dim = self.mnist_data[:,1:].shape
			except:
				raise Exception('Not able to parse MNIST Data')


		self.ncls = 10
		self.mode = mode
		self.batch_size = 128
		self.counter = 0
		self.nepochs = 10
		self.maxiters = 1000000
		self.lr = 1e-3
		self.eps = 1e-8
		self.save_model_path = model_path
		self.n_hidden = n_hidden

		self.weights={}
		self.biases={}
		self.batch_norm_params={}
		self.bn_cache={}
		self.save_iter = 100

		if mode == 'train':
			print "Initializing Weights......"
			self.init_weights(n_hidden,hidden_dims)
		else:
			print "Loading Trained Weights......"
			self.load_weight(model_path)

	def load_weight(self,path):
		if path is None:
			raise Exception('Model Load Path Invalid')
		try:	
			self.weights = np.load(os.path.join(path,'weights.npy')).item()
		except:
			raise Exception('Not able to load Network weights')

		try:	
			self.biases = np.load(os.path.join(path,'biases.npy')).item()
		except:
			raise Exception('Not able to load Network biases')

		try:	
			self.batch_norm_params = np.load(os.path.join(path,'bn_params.npy')).item()
		except:
			raise Exception('Not able to load Network weights')


	def init_weights(self,n_hidden,dims):
		self.weights[0]=np.random.randn(self.ip_dim,dims[0]) * (1.0/np.sqrt(self.ip_dim))
		self.biases[0] = np.zeros((dims[0],))

		for i in range(1,n_hidden):
			self.weights[i] = np.random.randn(dims[i-1],dims[i])*(1.0/np.sqrt(dims[i-1]))
			self.biases[i] = np.zeros((dims[i],))

		self.weights[-1] = np.random.randn(dims[n_hidden-1],self.ncls)*(1.0/np.sqrt(dims[n_hidden-1]))
		self.biases[-1] = np.zeros((self.ncls,))

		for i in range(n_hidden):
			self.batch_norm_params[i] = {'gamma':np.ones((dims[i],)),\
										'beta':np.zeros((dims[i],)), \
										'running_mean':np.zeros((dims[i],)),\
										'running_var':np.zeros((dims[i],)),}
			self.bn_cache[i]=[]


	def get_data(self):
		
		if self.counter > self.n_samples:
			self.counter = 0

		if(self.counter == 0):
			np.random.shuffle(self.mnist_data)
		return self.mnist_data[self.counter:self.counter+self.batch_size,:]

	def forward(self,ip,labels=None):
		h1 = ip.dot(self.weights[0]) + self.biases[0]
		h1_bn = self.batchnorm_forward(h1,0,mode=self.mode)
		h1_relu = self.relu(h1_bn.copy())
		
		h2 = h1_relu.dot(self.weights[1]) + self.biases[1]
		h2_bn = self.batchnorm_forward(h2,1,mode=self.mode)
		h2_relu = self.relu(h2_bn.copy())

		
		logits = h2_relu.dot(self.weights[-1]) + self.biases[-1]
		probs = self.softmax(logits)
		if labels is not None:
			loss = self.loss(labels,probs)
		else:
			loss = None
		return (h1,h1_bn,h1_relu,h2,h2_bn,h2_relu,logits,probs,loss)

	def loss(self,y,pred):
		return -np.sum(y*np.log(pred))/(y.shape[0]+self.eps)

	def softmax(self,ip):
		centered_ip = ip - np.max(ip,axis=1).reshape(-1,1)
		probs = np.exp(centered_ip)
		return probs/np.sum(probs,axis=1).reshape(-1,1).astype(np.float32)

	def relu(self,ip):
		x = ip
		x[np.where(x < 0)] = 0
		return x

	def batchnorm_forward(self,ip,index,mode='train'):
		'''
		Perform batch normalization over ip (Nx D)
		'''
		gamma = self.batch_norm_params[index]['gamma']
		beta = self.batch_norm_params[index]['beta']
		running_mean = self.batch_norm_params[index]['running_mean']
		running_var = self.batch_norm_params[index]['running_var']
		N,D = ip.shape
		if mode == 'train':

			x = ip
			mu = np.mean(x,axis=0) # D,
			xmu = x - mu
			
			delta = (x-mu)**2
			var = (np.mean(delta,axis=0))
			sqrtvar = np.sqrt(var+self.eps) 
			invvar = 1/sqrtvar

			x_hat = xmu*invvar
			x_hat_gamma = x_hat*gamma
			out = x_hat_gamma + beta

			running_mean = 0.9*running_mean + 0.1*mu
			running_var = 0.9*running_var + 0.1*var
			self.batch_norm_params[index]['running_mean'] = running_mean
			self.batch_norm_params[index]['running_var'] = running_var
			self.bn_cache[index] = [x,mu,xmu,delta,var,sqrtvar,invvar,x_hat,x_hat_gamma]
		else:
			x = ip
			xmu = x - running_mean
			x_hat = xmu/np.sqrt(running_var+self.eps)
			out = x_hat*gamma + beta

		return out

	def batchnorm_backward(self,din,index):
		x,mu,xmu,delta,var,sqrtvar,invvar,x_hat,x_hat_gamma = self.bn_cache[index]
		gamma = self.batch_norm_params[index]['gamma']
		beta = self.batch_norm_params[index]['beta']

		N,D = din.shape

		dbeta = np.sum(din,axis=0)
		dgamma = np.sum(din*x_hat,axis=0)

		dx_1 = gamma*invvar*din # del
		dx_2 = -gamma*invvar*np.sum(din,axis=0)/N
		dx_3 = -gamma*(xmu)*(var+self.eps)**(-1.5)*(np.sum(din*(xmu),axis=0))/float(N)

		dx = dx_1+dx_2+dx_3
		return dx,dbeta,dgamma


	def backward(self,cache,ip,labels):
		h1,h1_bn,h1_relu, h2,h2_bn,h2_relu, logits, probs,__ = cache
		dy = (labels - probs)
		dW2 = h2_relu.T.dot(dy)
		db2 = np.sum(dy,axis=0)

		dh2 = dy.dot(self.weights[-1].T)
		dh2[np.where(h2 == 0)] = 0
		dh2, dbeta2,dgamma2 = self.batchnorm_backward(dh2,1)

		dW1 = h1_relu.T.dot(dh2)
		db1 = np.sum(dh2,axis=0)

		dh1 = dh2.dot(self.weights[1].T)
		dh1[np.where(h1 == 0)] = 0
		dh1, dbeta1,dgamma1 = self.batchnorm_backward(dh1,0)

		dW0 = ip.T.dot(dh1)
		db0 = np.sum(dh1,axis=0)

		return (dW0,db0,dW1,db1,dW2,db2,dbeta2,dgamma2,dbeta1,dgamma1)

	def update(self,grads):
		
		dW0,db0,dW1,db1,dW2,db2,dbeta2,dgamma2,dbeta1,dgamma1 = grads

		self.weights[0] += self.lr*(dW0)
		self.biases[0] += self.lr*(db0)
		
		self.weights[1] += self.lr*(dW1)
		self.biases[1] += self.lr*(db1)

		self.weights[-1] += self.lr*(dW2)
		self.biases[-1] += self.lr*(db2)

		self.batch_norm_params[0]['gamma'] += self.lr*dgamma1
		self.batch_norm_params[1]['gamma'] += self.lr*dgamma2

		self.batch_norm_params[0]['beta'] += self.lr*dbeta1
		self.batch_norm_params[1]['beta'] += self.lr*dbeta2


	def accuracy(self,preds,labels):
		p_idx = np.argmax(preds,axis=1)
		l_idx = np.argmax(labels,axis=1)

		pos = len(np.where(p_idx-l_idx == 0)[0])
		return pos/(float(preds.shape[0])+self.eps)


	def train(self):
		it_per_epoch = self.n_samples/self.batch_size + 1
		it_num = 0
		acc_cache =[]
		loss_cache=[]
		it_cache=[]
		print "Starting Training.............."
		

		for i in range(self.nepochs):
			for j in range(it_per_epoch):

				batch_data = self.get_data()
				self.counter += self.batch_size

				batch_imgs = batch_data[:,1:].astype(np.float32)/255.0 - 0.5
				batch_labels = np.zeros((batch_imgs.shape[0],self.ncls))
				batch_labels[np.arange(batch_labels.shape[0]),batch_data[:,0].astype(np.uint8).tolist()] = 1

				cache = self.forward(batch_imgs,batch_labels)
				
				grads = self.backward(cache,batch_imgs,batch_labels)

				self.update(grads)
				if not it_num%self.save_iter:
					new_acc = self.accuracy(cache[-2],batch_labels)
					new_loss = cache[-1]
					acc_cache.append(new_acc)
					loss_cache.append(cache[-1])
					it_cache.append(it_num)
					print "Training Iteration===>%d, Loss ====>%.4f, Acc ====>%.4f"%(it_num,new_loss,new_acc)
				it_num +=1
				
				if it_num > self.maxiters:
					break
			if it_num > self.maxiters:
				break

		print "Done Training!"
		self.save_model()
		print "Model Saved!"
		plt.plot(it_cache,loss_cache)
		plt.xlabel('Iterations')
		plt.ylabel('Loss')
		plt.show()

		plt.plot(it_cache,acc_cache)
		plt.xlabel('Iterations')
		plt.ylabel('Accuracy')
		plt.show()


	def test(self):
		it_per_epoch = self.n_samples/self.batch_size + 1
		it_num = 0
		for j in range(it_per_epoch):

			batch_data = self.get_data()
			self.counter += self.batch_size

			batch_imgs = batch_data[:,1:].astype(np.float32)/255.0 - 0.5
			batch_labels = np.zeros((batch_imgs.shape[0],self.ncls))
			batch_labels[np.arange(batch_labels.shape[0]),batch_data[:,0].astype(np.uint8).tolist()] = 1

			cache = self.forward(batch_imgs,batch_labels)
			print "Testing Iteration===>%d, Acc ====>%.4f"%(it_num,self.accuracy(cache[-2],batch_labels))
			it_num +=1
			

	def save_model(self):
		if not os.path.exists(self.save_model_path):
			os.makedirs(self.save_model_path)
		else:
			map(lambda x: os.unlink(os.path.join(self.save_model_path,x)), os.listdir(self.save_model_path))

		np.save(os.path.join(self.save_model_path,'bn_params.npy'),self.batch_norm_params)
		np.save(os.path.join(self.save_model_path,'weights.npy'),self.weights)
		np.save(os.path.join(self.save_model_path,'biases.npy'),self.biases)




if __name__ == '__main__':
	
	mode = sys.argv[1]
	dpath = sys.argv[2]
	model_path = sys.argv[3]
	
	obj = NN(mode=mode,dpath=dpath,model_path=model_path)
	
	if mode == 'train':
		obj.train()
	else:
		obj.test()
	










