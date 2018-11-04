

from neural_network import *





def train_csv(nn, columns,results,train_data,delimiter=","):
	size = 200
	success = 0.0
	learn_rate = 0.4

	df_train = pd.read_csv(train_data,delimiter)

	for i, row in df_train.iterrows():
		x = []
		real = []
		for c in columns:
			x.append(row[c])
		for r in results:
			real.append(row[r])

		'''
		exp_out = np.zeros(10)
		exp_out[math.floor(real[0])] = 1.0              #asd
		'''
		exp_out= real

		res, outputs = nn.forward_feeding(x)
		deltam = nn.error_backpropagation(outputs, exp_out)
		nn.upgrade_wb(deltam, x, learn_rate, outputs)

		success += ((res[0] > 0.5)==real)
		'''
		succ = 1.0-np.linalg.norm(exp_out-res)
		if(succ > 0.0):
			success += succ
		'''
		if (i % size == 0):
			ratio= (success/size)
			success=0.0
			print(i,"\t",ratio)
	print(nn.to_str())
	return nn


def train_csv2(nn, columns,results,train_data,delimiter=","):
	size = 200
	success = 0.0
	learn_rate = 0.4

	df_train = pd.read_csv(train_data,delimiter)

	for i, row in df_train.iterrows():
		x = []
		real = []
		for c in columns:
			x.append(row[c])
		for r in results:
			real.append(row[r])

		res, outputs = nn.forward_feeding(x)
		print("real",real)
		print("res",res)

		deltam = nn.error_backpropagation(outputs, real)
		nn.upgrade_wb(deltam, x, learn_rate, outputs)

		#success += ((res[0] > 0.5)==real)
		#print(exp_out)
		#print(res)
		succ = 1.0-np.linalg.norm(np.array(real)-res)
		if(succ > 0.0):
			success += succ
		#print(succ)
		if (i % size == 0):
			ratio= (success/size)
			success=0.0
			print(i,"\t",ratio)
	return nn



def test_csv(nn,columns,results,test_data,delimiter=","):


	xreal= xpred=[]
	df_test = pd.read_csv(test_data,delimiter)

	for i, row in df_test.iterrows():
		x = []
		real = []
		for c in columns:
			x.append(row[c])
		for r in results:
			real.append(row[r])

		res, outputs = nn.forward_feeding(x)

		xreal.append(real[0])
		xpred.append(res[0] > 0.5)

	print("precision:\t", precision_score(xreal, xpred, ))
	print("recall:   \t", recall_score(xreal, xpred))



columns = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio',]
results = ['Occupancy',]

layers = make_layers([5, 4, 1])
nn = NeuralNetwork(layers)

nn = train_csv(nn,columns,results,'data/datatraining.txt')
test_csv(nn, columns, results, 'data/datatest.txt')
test_csv(nn, columns, results, 'data/datatest2.txt')




'''
expected_output= np.array( [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,])
output          =  [0.008229692143327479, 0.00812664162522944, 0.008201673882462013, 0.009488489621260287, 0.02348916443033424, 0.14106864538762823, 0.5994561066547094, 0.18384794381847558, 0.02283062963625901, 0.008340408937136655]

print(expected_output)
print(output)


print(np.linalg.norm(expected_output - expected_output))
print(np.linalg.norm(expected_output - output))
output          =  [0.008229692143327479, 0.00812664162522944, 0.008201673882462013, 0.009488489621260287, 0.02348916443033424, 0.14106864538762823, 0.9994561066547094, 0.18384794381847558, 0.02283062963625901, 0.008340408937136655]
print(np.linalg.norm(expected_output - output))







precision = tp / (tp + fp)
recall = tp / (tp + fn)

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





columns = ['fixed acidity',"volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol",]
results = ["quality",]
delimiter = ";"

layers = make_layers([11,8,4,10])
nn = NeuralNetwork(layers)

nn = train_csv(nn,columns,results,'data/winequality-white.csv',delimiter)
#test_csv(nn, columns, results, 'data/winequality-white.csv',delimiter)

'''



'''
def train_occupancy():
	size = 200
	success = 0.0

	learn_rate = 0.4
	layers = make_layers([5,4,1])
	nn = NeuralNetwork(layers)
	df_train = pd.read_csv('data/datatraining.txt')

	for i, row in df_train.iterrows():
		x = [
			#row["date"],
			row["Temperature"],
			row["Humidity"],
			row["Light"],
			row["CO2"],
			row["HumidityRatio"],

		]
		real = row["Occupancy"]

		res, outputs = nn.forward_feeding(x)
		deltam = nn.error_backpropagation(outputs, [real])
		nn.upgrade_wb(deltam, x, learn_rate, outputs)

		success += ((res[0] > 0.5)==real)
		if (i % size == 0):
			ratio= (success/size)
			success=0.0
			print(i,"\t",ratio)
	return nn

def test_occupancy():
	nn = train_occupancy()
	print(nn.to_str())

	xreal= xpred=[]
	df_test = pd.read_csv('data/datatest.txt')

	for i, row in df_test.iterrows():
		x = [
			#row["date"],
			row["Temperature"],
			row["Humidity"],
			row["Light"],
			row["CO2"],
			row["HumidityRatio"],
		]
		real = row["Occupancy"]
		res, outputs = nn.forward_feeding(x)

		xreal.append(real)
		xpred.append(res[0] > 0.5)

	print("precision:\t", precision_score(xreal, xpred, ))
	print("recall:   \t", recall_score(xreal, xpred))

test_occupancy()
'''
