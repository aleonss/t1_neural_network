from neural_network import *

import pandas as pd

import time


def train_nn(nn, learn_rate, columns, results, df_train):
	print("\nTraining:",len(df_train),"rows")
	size = round(len(df_train) / 20)
	error = 0.0
	iters = 0
	df_train = df_train.iloc[np.random.permutation(len(df_train))]

	for i, row in df_train.iterrows():
		input = []
		real = []
		for c in columns:
			input.append(row[c])
		for r in results:
			real.append(row[r])
		exp_out= real

		res, outputs = nn.forward_feeding(input)
		deltam = nn.error_backpropagation(outputs, exp_out)
		nn.upgrade_wb(deltam, input, learn_rate, outputs)

		error += abs(real - res[0])
		if( (iters % size == 0) and (iters!=0) ):
			ratio= (error/iters)
			error=0.0
			print(iters,"\t error:",ratio[0])
		iters+=1

	return nn


def test_nn(nn, columns, results, df_test):
	print("\nTesting:",len(df_test),"rows")

	xreal=[]
	xpred=[]
	df_test = df_test.iloc[np.random.permutation(len(df_test))]

	for i, row in df_test.iterrows():
		inputs = []
		real = []
		for c in columns:
			inputs.append(row[c])
		for r in results:
			real.append(row[r])
		res, outputs = nn.forward_feeding(inputs)

		xreal.append(real[0]==1)
		xpred.append(res[0] > 0.5 )

	get_performance(xreal, xpred)





def datetime_to_time(dataframe):
	dataframe['time'] = dataframe['date'].str.split(' ').str.get(1).str.split(':')
	dataframe['hour'] = pd.to_numeric(dataframe['time'].str.get(0))
	dataframe['minute'] = pd.to_numeric(dataframe['time'].str.get(1))
	return dataframe



def occupancy_prediction(learn_rate, layers):
	columns = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio',] #'hour', 'minute']
	results = ['Occupancy', ]

	nn = NeuralNetwork(layers)
	#data/datatraining.txt       #8144
	#data/datatest.txt           #2666
	#data/datatest2.txt          #9753

	df_train = pd.read_csv('data/datatraining.txt') #datetime_to_time(pd.read_csv('data/datatraining.txt', ))
	df_test1 = pd.read_csv('data/datatest.txt')  #datetime_to_time(pd.read_csv('data/datatest.txt', ))
	df_test2 = pd.read_csv('data/datatest2.txt') #datetime_to_time(pd.read_csv('data/datatest2.txt', ))

	t0 = time.time()
	nn = train_nn(nn, learn_rate, columns, results, df_test1)
	t1 = time.time()
	ttrain = (t1 - t0)*1000.0
	t0 = time.time()
	test_nn(nn, columns, results, df_train)
	t1 = time.time()
	ttest0 = (t1 - t0)*1000.0
	t0 = time.time()
	test_nn(nn, columns, results, df_test2)
	t1 = time.time()
	ttest1 = (t1 - t0)*1000.0

	print("Time::")
	print("\tTraining:",ttrain,"ms")
	print("\tTest0:",ttest0,"ms")
	print("\tTest1:",ttest1,"ms")

	print("\n\nNeural Network::")
	print(nn.to_str())


layers = make_layers([5, 16, 8, 4, 1])
learn_rate = 0.5
occupancy_prediction(learn_rate,layers)




def test_learn_rates():
	layers = make_layers([5, 16, 8, 4, 1])
	for i in range(1,9):
		learn_rate = i/10
		print(" --Learn Rate :",learn_rate,"--")
		occupancy_prediction(learn_rate,layers)


def test_layers():
	learn_rate = 0.5
	arr = [5]
	for i in range(2,9,2):
		for l in range(1,i,2):
			arr.append(round(i*4 / l))
		arr.append(1)
		print(" --Layers :",arr,"--")
		layers = make_layers(arr)
		occupancy_prediction(learn_rate,layers)
		arr = [5]

