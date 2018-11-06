from neural_network import *

import pandas as pd


def train_nn(nn,learn_rate, columns, results, df_train):
	size = 200
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

		error += abs(real -res[0])
		if( (iters % size == 0) and (iters!=0) ):
			ratio= (error/iters)
			error=0.0
			print(iters,"\t",ratio)
		iters+=1

	print(nn.to_str())
	return nn



def binary_classifiers(real,pred):
	tp = 0.0
	fn = 0.0
	fp = 0.0
	tn = 0.0
	for r,p in zip(real,pred):
		if(r and p):
			tp+=1
		elif(r and (not p)):
			fn+=1
		elif((not r) and p):
			fp+=1
		elif((not r) and (not p)):
			tn+=1
	return tp,fn,fp,tn

def get_performance(real, pred):
	tp, fn, fp, tn = binary_classifiers(real,pred)
	accuracy = (tp + tn) / (tp + tn + fp + fn)
	if(tp==0):
		precision = 0
		recall = 0
	else:
		precision = tp / (tp + fp)
		recall = tp / (tp + fn)

	print("Accuracy: ", accuracy)
	print("Precision:", precision )
	print("Recall:   ", recall , "\n")
	return accuracy,precision,recall



def test_nn(nn, columns, results, df_test):

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





columns = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio','hour','minute']
results = ['Occupancy',]

n_input = columns.__len__()
n_output = results.__len__()

learn_rate = 0.55
layers = make_layers([n_input,16,8,4,n_output])
nn = NeuralNetwork(layers)

'''
data/datatraining.txt       #8144
data/datatest.txt           #2666
data/datatest2.txt          #9753
'''

df_train = pd.read_csv('data/datatraining.txt',)
df_test1 = pd.read_csv('data/datatest.txt',)
df_test2 = pd.read_csv('data/datatest2.txt',)


def datetime_to_time(dataframe):
	dataframe['time'] = dataframe['date'].str.split(' ').str.get(1).str.split(':')
	dataframe['hour'] = pd.to_numeric(dataframe['time'].str.get(0))
	dataframe['minute'] = pd.to_numeric(dataframe['time'].str.get(1))
	return dataframe

df_train = datetime_to_time(df_train)
df_test1 = datetime_to_time(df_test1)
df_test2 = datetime_to_time(df_test2)


nn = train_nn(nn,learn_rate, columns, results, df_test1)
test_nn(nn, columns, results, df_train)
test_nn(nn, columns, results, df_test2)







'''
columns = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio',]
results = ['Occupancy',]

layers = make_layers([5,1])
nn = NeuralNetwork(layers)

df_train = pd.read_csv('data/datatraining.txt',)
df_test1 = pd.read_csv('data/datatest.txt',)
df_test2 = pd.read_csv('data/datatest2.txt',)

nn = train_csv(nn,columns,results,df_train)
test_csv(nn, columns, results, df_test1)
test_csv(nn, columns, results, df_test2)







Data Set Information:

Three data sets are submitted, for training and testing. Ground-truth occupancy was obtained from time stamped pictures that were taken every minute.
For the journal publication, the processing R scripts can be found in:
[Web Link]

Attribute Information:

date time year-month-day hour:minute:second
Temperature, in Celsius
Relative Humidity, %
Light, in Lux
CO2, in ppm
Humidity Ratio, Derived quantity from temperature and relative humidity, in kgwater-vapor/kg-air
Occupancy, 0 or 1, 0 for not occupied, 1 for occupied status





#WINE QUALITY

columns = ['fixed acidity',"volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol",]
results = ["quality",]
delimiter = ";"

layers = make_layers([11,8,4,10])
nn = NeuralNetwork(layers)

nn = train_csv(nn,columns,results,'data/winequality-white.csv',delimiter)
#test_csv(nn, columns, results, 'data/winequality-white.csv',delimiter)




#HANDWRITEN DIGIT RECOGNITION

columns = []
results = []

for i in range(0,256):
	columns.append("'p"+ i.__str__()+"'")
for i in range(0,10):
	results.append("'r"+ i.__str__()+"'")

layers = make_layers([256,32,10])
nn = NeuralNetwork(layers)

df_data = pd.read_csv('data/semeion2.data',delimiter=" ")
df_len = df_data.__len__()

df_train = df_data[0:int(df_len/2)]
df_test =  df_data[int(df_len/2):df_len]

nn = train_csv(nn,columns,results,df_train)
test_csv(nn, columns, results, df_test)
'''



'''
precision = tp / (tp + fp)
recall = tp / (tp + fn)

expected_output= np.array( [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,])
output          =  [0.008229692143327479, 0.00812664162522944, 0.008201673882462013, 0.009488489621260287, 0.02348916443033424, 0.14106864538762823, 0.5994561066547094, 0.18384794381847558, 0.02283062963625901, 0.008340408937136655]

print(expected_output)
print(output)


print(np.linalg.norm(expected_output - expected_output))
print(np.linalg.norm(expected_output - output))
output          =  [0.008229692143327479, 0.00812664162522944, 0.008201673882462013, 0.009488489621260287, 0.02348916443033424, 0.14106864538762823, 0.9994561066547094, 0.18384794381847558, 0.02283062963625901, 0.008340408937136655]
print(np.linalg.norm(expected_output - output))

'''


