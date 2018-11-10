from neural_network import *

import unittest
import numpy as np





def logic_or(x,y):
	return x or y

def logic_and(x,y):
	return x and y

def logic_xor(x,y):
	return np.logical_xor(x,y)


def logic_line(x,y):
	return x < y


def make_int_inputs(ninputs, in_range):
	inputs = []
	for i in range(0,ninputs):
		xx= random.randint(in_range[0],in_range[1])
		xy= random.randint(in_range[0],in_range[1])
		inputs.append([xx,xy])
	return inputs



def real_values(logic_funct,ninputs, in_range=[0,1]):
	inputs = make_int_inputs(ninputs, in_range)
	results = []
	for x in inputs:
		results.append(logic_funct(x[0],x[1]))
	return inputs,results





def nn_learn(nn, logic_funct, trainings = 1000, learn_rate = 0.5, in_range=[0,1], vervose=False ):
	if (vervose):
		print("Learning::")
		error = 0.0
		iters = 0
		size = trainings / 10

	inputs, results = real_values(logic_funct,trainings, in_range)

	for x,real in zip(inputs,results):
		res, outputs = nn.forward_feeding(x)
		deltam = nn.error_backpropagation(outputs, [real])
		nn.upgrade_wb(deltam, x, learn_rate, outputs)

		if(vervose):
			error += abs(real -res[0])
			if( (iters % size == 0) and (iters!=0) ):
				ratio= (error/iters)
				error=0.0
				print(iters,"\t error:",ratio)
			iters+=1

	return nn

def test_funct(sn,logic_funct, ntests=100, in_range=[0,1]):
	xreal=[]
	xpred=[]

	inputs, results = real_values(logic_funct,ntests, in_range)

	for x,real in zip(inputs,results):
		res = sn.feed(x)
		if (type(res) == list):
			result = sn.feed(x)[0] > 0.5
		else:
			result = sn.feed(x) > 0.5
		xreal.append(real)
		xpred.append(result)

	return xreal, xpred




class TestSigmoidNeuron(unittest.TestCase):
	def test_or(self):
		sn = SigmoidNeuron(weights= [3.76, 3.71], bias= -1.5)
		xreal, xpred = test_funct(sn, logic_or)
		for r, p in zip(xreal, xpred):
			self.assertEqual(r, p)


	def test_and(self):
		sn = SigmoidNeuron(weights= [2.91, 2.88], bias=-4.61)
		xreal, xpred = test_funct(sn, logic_and)
		for r, p in zip(xreal, xpred):
			self.assertEqual(r, p)


	def test_line(self):
		sn = SigmoidNeuron(weights= [-3.74,3.74], bias=-1.5)
		xreal, xpred = test_funct(sn, logic_line)
		for r, p in zip(xreal, xpred):
			self.assertEqual(r, p)





class TestNeuralNetwork(unittest.TestCase):
	def test_or(self):
		nn = NeuralNetwork([NeuronLayer(2, 1)])
		nn = nn_learn(nn, logic_or)

		xreal, xpred = test_funct(nn,logic_or)
		for r , p in zip(xreal,xpred):
			self.assertEqual(r, p)

	def test_and(self):
		nn = NeuralNetwork([NeuronLayer(2, 1)])
		nn = nn_learn(nn, logic_and)

		xreal, xpred = test_funct(nn,logic_and)
		for r , p in zip(xreal,xpred):
			self.assertEqual(r, p)

	'''
	def test_line(self):
		print("TestNeuron: line")
		nn = NeuralNetwork([NeuronLayer(2, 1)])
		nn = nn_learn(nn, logic_line, learn_rate=0.2, trainings = 50000, in_range=[-50,50])
		print("Neural Network::\n", nn.to_str())
		xreal, xpred = test_funct(nn,logic_line,in_range=[-50,50])
		for r , p in zip(xreal,xpred):
			self.assertEqual(r, p)
	'''

	def test_xor(self):
		#print("TestNeuron: xor")

		lay0 = NeuronLayer()
		lay1 = NeuronLayer()

		lay0.neurons =[
			SigmoidNeuron(weights= [4.1, -4.1],bias=- 2.2),
			SigmoidNeuron(weights=[-4.1, 4.1], bias=- 2.2),
		]
		lay1.neurons =[
			SigmoidNeuron(weights= [4.2, 4.2], bias=- 2.0),
		]

		nn = NeuralNetwork([lay0,lay1])
		nn = nn_learn(nn, logic_xor)

		xreal, xpred = test_funct(nn, logic_xor)
		for r , p in zip(xreal,xpred):
			self.assertEqual(r, p)



def test_line():
	print("\nTEST: LINE")
	nn = NeuralNetwork([NeuronLayer(2, 1)])
	nn = nn_learn(nn, logic_line, learn_rate=0.2, trainings=50000, in_range=[-50,50],vervose=True)
	print("Neural Network::\n",nn.to_str())

	xreal, xpred = test_funct(nn, logic_line, in_range=[-50,50])
	get_performance(xreal,xpred)
	inputs = make_int_inputs(100,[-50,50])
	plot_nn_2D(nn, 100, inputs, "line")


def test_and():
	print("\nTEST: AND")
	nn = NeuralNetwork([NeuronLayer(2, 1)])
	nn = nn_learn(nn, logic_and,vervose=True)
	print("Neural Network::\n",nn.to_str())

	xreal, xpred = test_funct(nn, logic_and)
	get_performance(xreal,xpred)
	inputs = make_int_inputs(100,[0,1])
	plot_nn_2D(nn, 100, inputs, "and")

def test_or():
	print("\nTEST: OR")
	nn = NeuralNetwork([NeuronLayer(2, 1)])
	nn = nn_learn(nn, logic_or,vervose=True)
	print("Neural Network::\n",nn.to_str())

	xreal, xpred = test_funct(nn, logic_or)
	get_performance(xreal,xpred)
	inputs = make_int_inputs(100,[0,1])
	plot_nn_2D(nn, 100, inputs, "or")


def test_xor():
	print("\nTEST: XOR")
	layers = make_layers([2,4,4,1])
	nn = NeuralNetwork(layers)
	nn = nn_learn(nn, logic_xor, learn_rate=0.2, trainings=30000, in_range=[0,1],vervose=True)
	print("Neural Network::\n",nn.to_str())

	xreal, xpred = test_funct(nn, logic_xor)
	get_performance(xreal,xpred)
	inputs = make_int_inputs(100,[0,1])
	plot_nn_2D(nn, 100, inputs, "xor")







if __name__ == '__main__':
	test_or()
	test_xor()
	test_line()
	test_and()
	unittest.main()



