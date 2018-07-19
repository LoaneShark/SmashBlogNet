import numpy as np 
import matplotlib.pyplot as plt 
import scipy as sp 
import csv
import math
from datetime import datetime,date,timedelta
import warnings
# NN imports
import pandas
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# fix random seed for reproducibility
np.random.seed(27)

verbosity = 2

# Smash Ultimate Blog Predictor
def main():
	# import data
	labels,data = readin()
	if verbosity >= 1:
		print("Analyzing %d fighters..." %(len(data)))

	for fighter in data:
		fighter[2] = workdays(fighter[2],date(2018,6,12))*1.0
	labels[2] = 'Weekdays since E3'
	
	names = data[:,0]
	# choose data columns we want
	data = data[:,1:] # ignore fighter names
	data = np.append(data[:,:2],data[:,3:],axis=1) # ignore days since E3
	labels = np.append(labels[1:3],labels[4:],axis=0)
	
	# split dataset into samples, timesteps, and features
	timesteps = data[:,1]
	samples = data[:,0]
	features = data[:,2:]
	
	#temp = np.append(timesteps.reshape(-1,1),features,axis=1)
	#data = np.append(samples.reshape(-1,1),temp,axis=1)
	labels = np.append([labels[0]],np.append([labels[1]],labels[2:]),axis=0)

	if verbosity >= 2:
		print(labels)
		print(data)

	# suppress warnings that my data started as ints and got converted to floats
	from sklearn.exceptions import DataConversionWarning
	warnings.filterwarnings(action='ignore', category=DataConversionWarning)
	# rescale all data to be within [0,1], except timesteps
	scaler = MinMaxScaler(feature_range=(0,1))
	samples_norm = scaler.fit_transform(samples.reshape(-1,1))
	feature_norm = scaler.fit_transform(features)

	# assemble database to be [samples,timesteps,features]
	dataraw = np.append(samples.reshape(-1,1), timesteps.reshape(-1,1),axis=1)
	dataset = np.zeros([len(dataraw),3],dtype='object')
	for i in range(len(dataraw)):
		dataset[i,:2] = dataraw[i]
		dataset[i,2] = feature_norm[i]

	if verbosity >= 3:
		print("PROCESSED DATASET:\n",dataset)
		print(dataset.shape)

	# split into train & test sets
	# NOTE: should they be split differently than this?
	train_size = int(len(dataset)*0.667)
	test_size = len(dataset)-train_size
	train,test = dataset[:train_size,:], dataset[train_size:,:]
	if verbosity >= 1:
		print("Train size: " + str(len(train)))
		print("Test size: " + str(len(test)))

	trainX, trainY = create_dataset(train, 1)
	testX, testY = create_dataset(test, 1)
	#trainX = trainX[:,0] # clean up data for some reason
	#testX = testX[:,0]
	if verbosity >= 2:
		print("testX: \n", trainX)
		print("testY: \n", testY)
	
	model_dataset(trainX,trainY,testX,testY)

# create and train the LSTM network
def model_dataset(trainX,trainY,testX,testY,look_back=1):

	model = Sequential()
	model.add(LSTM(4, input_shape=(3,look_back)))
	model.add(Dense(3))
	model.compile(loss='mean_squared_error',optimizer='adam')
	model.fit(trainX,trainY,epochs=100,batch_size=1,verbose=2)

	# make predictions
	trainPredict = model.predict(trainX)
	testPredict = model.predict(testX)
	# invert predictions
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY = scaler.inverse_transform([trainY])
	testPredict = scaler.inverse_transform(testPredict)
	testY = scaler.inverse_transform([testY])
	# calculate root mean squared error
	trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
	print('Train Score: %.2f RMSE' % (trainScore))
	testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
	print('Test Score: %.2f RMSE' % (testScore))

	# shift train predictions for plotting
	trainPredictPlot = numpy.empty_like(dataset)
	trainPredictPlot[:, :] = numpy.nan
	trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
	# shift test predictions for plotting
	testPredictPlot = numpy.empty_like(dataset)
	testPredictPlot[:, :] = numpy.nan
	testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
	# plot baseline and predictions
	plt.plot(scaler.inverse_transform(dataset))
	plt.plot(trainPredictPlot)
	plt.plot(testPredictPlot)
	plt.show()


def create_dataset(dataset, look_back=1):
	dataX, dataY = [],[]

	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back),:]

		dataX.append(a)
		dataY.append(dataset[i + look_back, :])
	return np.array(dataX), np.array(dataY)

# reads in specified data csv and returns a
# 	numpy array containing the cleaned data
# 	and another containing the labels
def readin(filepath='blogdata.csv'):
	data = []

	# open file
	with open(filepath) as csvfile:
		datareader = csv.reader(csvfile)
		for fighter in datareader:
			data.append(fighter)

	labels = data[0]
	n = len(labels)
	data = data[1:]

	# ready to use data 
	data = parsecsv(data,n)
	return np.array(labels),data

# cleans CSV and assigns proper data types
# returns numpy array of data vectors, without labels
def parsecsv(data,n):
	cleandata = np.zeros([1,n])
	_date = lambda x: datetime.strptime(x, "%m/%d/%y").date()
	parsers = [str,int,_date,int,int,int,int,int,float]

	for fighter in data:
		parsed = [parse(inp) for parse,inp in zip(parsers,fighter)]
		cleandata = np.append(cleandata,[parsed],axis=0)

	return cleandata[1:]

# returns the number of WEEKDAYS between two dates
def workdays(finaldate, refdate = date(2018,6,12)):
	daygen = (refdate + timedelta(x + 1) for x in range((finaldate-refdate).days))
	return sum(1 for day in daygen if day.weekday()<5)


if __name__ == "__main__":
	main()