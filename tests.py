from neural_network import *



class TestNeuron(unittest.TestCase):

	def learn_or(self):
		trainings = 1000
		lr = 0.5

		sn = SigmoidNeuron()
		sn.mrand(2)

		for t in range(0, trainings):
			# success=0.0
			for i in range(0, 2):
				for j in range(0, 2):
					x = [i, j]
					real = i or j
					res = sn.feed(x) > 0.5
					diff = abs(real - res)
					for k in range(0, 2):
						sn.weights[k] += (lr * diff * x[k])

			# success += (1.0 - diff)
		# if (t % 10 == 0):
		#	ratio= (success/(4+0.0))
		#	print(t,"\t",ratio)
		return sn

	def test_or(self):
		sn = self.learn_or()
		print("weights:\t",sn.weights)
		print("bias:   \t",sn.bias)
		for i in range(0,2):
			for j in range(0,2):
				x=[i,j]
				real = i or j
				result = sn.feed(x) > 0.5
				self.assertEqual(real, result)
				#print(x,"\treal:",real,"\tres:",result)

	'''
	def test_and(self):
		w = [1.0,1.0]
		b = -1.5
		sn = SigmoidNeuron(w, b)

		for i in range(0,2):
			for j in range(0,2):
				x=[i,j]
				real = i and j
				result = sn.feed(x)
				self.assertEqual(real, result)

	def test_not(self):
		w = [-1.0]
		b = 0.5
		sn = SigmoidNeuron(w, b)

		for i in range(0,2):
			x=[i]
			real = not i
			result = sn.feed(x)
			self.assertEqual(real, result)

	def test_nand(self):
		w = [-2.0,-2.0]
		b = 3
		sn = SigmoidNeuron(w, b)

		for i in range(0,2):
			for j in range(0,2):
				x=[i,j]
				real = not(i and j)
				result = sn.feed(x)
				self.assertEqual(real, result)

	def test_sum(self):
		for i in range(0,2):
			for j in range(0,2):

				real_summ = (i+j)%2
				real_carr = (i+j)>1

				res_summ, res_carr = sum_nand(i,j)

				self.assertEqual(real_summ, res_summ)
				self.assertEqual(real_carr, res_carr)
	'''


def logical_inputs(size):
	inputs = []
	for i in range(0,size):
		xx= random.randint(0,1)
		xy= random.randint(0,1)
		inputs.append([xx,xy])
	return inputs


def real_xor(n_input):
	inputs = []
	results = []
	for i in range(0,n_input):
		xx= random.randint(0,1)
		xy= random.randint(0,1)

		inputs.append([xx,xy])
		results.append(np.logical_xor(xx,xy))

	return inputs,results

def real_line(n_input):
	inputs = []
	results = []
	for i in range(0,n_input):
		xx= random.uniform(-50.0, 50.0)+0.0
		xy= random.uniform(-50.0, 50.0)

		inputs.append([xx,xy])
		results.append(0.0+xx<xy)

	return inputs,results


def learn_or():
	size = 4
	trainings= 1000
	learn_rate = 0.4
	layers = make_layers([2,4,2,1])

	nn = NeuralNetwork(layers)

	for t in range(0,trainings):
		success = 0.0

		for i in range(0,2):
			for j in range(0,2):

				x=[i,j]
				real = i or j
				res, outputs = nn.forward_feeding(x)

				deltam = nn.error_backpropagation(outputs, [real])
				nn.upgrade_wb(deltam, x, learn_rate,outputs)

				result = res[0] > 0.5
				success += (1.0 - abs(real - result))

		if (t % 10 == 0):
			ratio= (success/(size+0.0))
			print(t,"\t",ratio)
	return nn


def train_xor():
	size = 4
	trainings= 1000
	learn_rate = 0.4
	layers = make_layers([2,10,1])

	nn = NeuralNetwork(layers)

	for t in range(0,trainings):
		success = 0.0
		for s in range(0,size):
			i = random.randint(0,1)
			j = random.randint(0,1)
			x=[i,j]
			real = np.logical_xor(i,j)+0.0

			res, outputs = nn.forward_feeding(x)
			deltam = nn.error_backpropagation(outputs, [real])
			nn.upgrade_wb(deltam, x, learn_rate,outputs)

			result = res[0] > 0.5
			success += (1.0 - abs(real - result))


		if (t % 10 == 0):
			ratio= (success/(size+0.0))
			print(t,"\t",ratio)

	return nn





def train_line():
	size = 4
	trainings= 1000
	learn_rate = 0.4

	layers = make_layers([2,10,5,1])
	nn = NeuralNetwork(layers)

	success = 0.0
	for t in range(0,trainings):
		xx= random.uniform(-50.0, 50.0)+0.0
		xy= random.uniform(-50.0, 50.0)

		x = [xx,xy]
		real = 0.0+xx<xy

		res, outputs = nn.forward_feeding(x)
		deltam = nn.error_backpropagation(outputs, [real])
		nn.upgrade_wb(deltam, x, learn_rate, outputs)

		success += (1.0 - abs(real - res[0]))
		if (t % 100 == 0):
			ratio= (success/100)
			success=0.0
			print(t,"\t",ratio)

	return nn




def test_line():
	nn =train_line()
	print(nn.to_str())

	size=100
	x, dOutput = real_line(size)
	plot_nn_2D(nn, size, x, "line")


def test_or():
	nn = learn_or()
	print(nn.to_str())

	for i in range(0,2):
		for j in range(0,2):
			x=[i,j]
			real = i or j
			res, outputs = nn.forward_feeding(x)
			result = res[0] > 0.5
			print(x,"\treal:",real,"\tres:",result)
	size = 100
	x = logical_inputs(size)
	plot_nn_2D(nn, size, x, "or")

def test_xor():
	nn = train_xor()
	print(nn.to_str())

	size = 100
	xreal= xpred=[]
	for s in range(0, size):

		i = random.randint(0, 1)
		j = random.randint(0, 1)
		x = [i, j]

		real = np.logical_xor(i, j)
		res, outputs = nn.forward_feeding(x)

		xreal.append(real)
		xpred.append(res[0] > 0.5)

	print("precision:\t", precision_score(xreal,xpred,))
	print("recall:   \t", recall_score(xreal,xpred))
	x = logical_inputs(size)
	plot_nn_2D(nn, size, x, "xor")





if __name__ == '__main__':
	test_or()
	test_line()
	test_xor()
	unittest.main()



