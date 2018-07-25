import numpy as np 
import matplotlib.pyplot as plt 
import scipy as sp 
import csv
import math
from datetime import datetime,date,timedelta
import warnings
# NN imports
import pandas
from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.exceptions import NotFittedError
# fix random seed for reproducibility
#np.random.seed(27)

verbosity = 1

# Smash Ultimate Blog Predictor
def main():
	# import data
	labels,data = readin()
	if verbosity >= 1:
		print("Analyzing %d fighters..." %(len(data)))

	# clean up data and add in missing info
	last = 0
	blankday = np.array(["None",-1,None,-1,"None","None","None","None",-1,"None","None"])

	cleandata = np.zeros((1,data.shape[1]),dtype='object')
	cleandata[0] = blankday
	
	# calculate weekday count for each post
	# add in missing days with blankday placeholders
	today = date.today()
	for fighter in data:
		fighter[2] = workdays(fighter[2],date(2018,6,12))*1.0
		diff = fighter[2] - last
		while diff >= 2:
			diff = diff-1
			tempday = np.copy(blankday)
			tempday[2] = fighter[2] - diff
			#print(tempday.shape)
			cleandata = np.append(cleandata,[tempday],axis=0)
		cleandata = np.append(cleandata,[fighter],axis=0)
		last = fighter[2]
	last = cleandata[-1,2]
	padcount = int(workdays(today)-last)
	print(last, padcount)
	for i in range(1,padcount):
		tempday = np.copy(blankday)
		tempday[2] = last+i
		cleandata = np.append(cleandata,[tempday],axis=0)

	data = cleandata[1:]
	if verbosity >= 3:
		print(data.shape,data)

	labels[2] = 'Weekdays since E3'
	# establish mapping for labels to their index
	labelkeys = {}
	for i in range(len(labels)):
		labelkeys[labels[i]] = i
	
	# select just the data columns we want, and reorder them
	titles = ['Number','Type','Series','Game Added','Game Count','3rd Party?','Newcomer?','Returning Vet?']
	features = np.array([data[:,labelkeys[title]] for title in titles],dtype='object')
	features = np.array([features[:,i] for i in range(len(data))])
	
	if verbosity >= 4:
		print(titles,"\n",features)

	labels = titles
	if verbosity >= 3:
		print(labels)
		print(data)
	dataset = np.copy(features)

	# encode nominal data to numerical values
	dataset,encoders = encode_dataset(dataset,titles)
	n_features = features.shape[1]

	# suppress warnings that my data started as ints and got converted to floats
	from sklearn.exceptions import DataConversionWarning
	warnings.filterwarnings(action='ignore', category=DataConversionWarning)
	# normalize dataset so all features are within [0,1]
	scaler = MinMaxScaler(feature_range=(0,1))
	scaled = scaler.fit_transform(dataset)
	# save last fighter for predicting next one
	recent = scaled[-1]

	# frame as supervised learning
	dataset = series_to_supervised(scaled,1,1)

	# drop columns that don't matter
	# (everything except fighter number for output)
	dataset.drop(dataset.columns[range(n_features+1,2*n_features)],axis=1, inplace=True)
	#print(dataset.head())

	if verbosity >= 2:
		print("PROCESSED DATASET:\n",dataset)
		print(dataset.shape)

	# split into train & test sets
	# NOTE: should they be split differently than this?
	values = dataset.values
	train_size = int(len(values)*0.667)
	test_size = len(values)-train_size
	train,test = values[:train_size,:], values[train_size:,:]
	if verbosity >= 1:
		print("Train size: " + str(len(train)))
		print("Test size: " + str(len(test)))

	# organize into [samples,timesteps,features] like keras wants
	trainX, trainY = create_dataset(train, 1)
	testX, testY = create_dataset(test, 1)

	# separate inputs and outputs
	trainX = trainX[:,:,:-1]
	testX = testX[:,:,:-1]
	trainY = trainY.reshape(-1)
	testY = testY.reshape(-1)
	if verbosity >= 2:
		print("trainX: ",trainX.shape,"\n",trainX)
		print("trainY: ",trainY.shape,"\n",trainY)
	
	# model LSTM network
	model,hist = model_dataset(trainX,trainY,testX,testY)

	# make predictions!
	model_predict(model,testX,testY,recent.reshape(1,1,n_features),scaler,encoders,titles)

# create and train the LSTM network
def model_dataset(trainX,trainY,testX,testY,look_back=1,plot_results=True):
	if verbosity >= 1:
		print("Generating model")

	n_feats = trainX.shape[2]

	# generate and train network
	model = Sequential()
	model.add(LSTM(20, input_shape=(look_back,n_feats)))
	model.add(Dense(1))
	model.compile(loss='mae',optimizer='adam')
	history = model.fit(trainX,trainY,epochs=250,batch_size=1,validation_data=(testX,testY),verbose=0,shuffle=False)

	# plot
	if plot_results:
		plt.plot(history.history['loss'],label='train')
		plt.plot(history.history['val_loss'], label='test')
		plt.legend()
		plt.show()

	return model,history

def model_predict(model,testX,testY,mostrecent,scaler,encoders,titles):

	# calculate RSME
	yhat = model.predict(testX,batch_size=1)
	testX = testX.reshape((testX.shape[0],testX.shape[2]))
	inv_yhat = np.concatenate((yhat, testX[:,1:]),axis=1)
	inv_yhat = scaler.inverse_transform(inv_yhat)[:,0]

	testY = testY.reshape((len(testY),1))
	inv_y = np.concatenate((testY,testX[:,1:]),axis=1)
	inv_y = scaler.inverse_transform(inv_y)[:,0]

	rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
	print('Test RMSE: %.3f' %rmse)

	# predict next fighter
	predicted = model.predict(mostrecent,batch_size=1,verbose=1)
	#print(predicted)

	mostrecent = scaler.inverse_transform(mostrecent[0])
	last = np.zeros(len(titles),dtype='object')
	for i in range(len(titles)):
		feature = titles[i]
		encoder = encoders[i]
		#print(i,": ",feature)

		# suppress warnings caused by sklearn bug
		warnings.filterwarnings(action='ignore',category=DeprecationWarning)
		try:
			last[i] = encoder.inverse_transform(mostrecent[0,i].astype(int))
		except NotFittedError:
			last[i] = mostrecent[0,i].astype(int)

	#mostrecent = mostrecent.reshape((len(mostrecent),1))
	print("Last Posted Fighter: \n",titles)
	print(" ",last)

	pred_num = predicted[:,0]*66
	pred_num = pred_num.astype(int)
	
	print("Next Predicted Fighter Number(s): ",pred_num)
	return rmse,pred_num

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols,names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def encode_dataset(data,labels):
	if verbosity >= 1:
		print("Encoding dataset...")

	encoders = [LabelEncoder() for i in range(len(labels))]
	dataset = np.copy(data)
	for i in range(len(labels)):

		item = data[0,i]
		if type(item) is not float:
			if type(item) is int:
				dataset[:,i] = data[:,i].astype('float32')
			else:
				dataset[:,i] = encoders[i].fit_transform(data[:,i])
	dataset.astype('float32')
	return dataset, encoders

def create_dataset(dataset, look_back=1):
	dataX, dataY = [],[]

	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back),:]

		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY).reshape(1,len(dataY))

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
	_bool = lambda x: True if x == "True" else False
	#parsers = [str,int,_date,int,str,_bool,_bool,_bool,int,str,str]
	parsers = [str,int,_date,int,str,str,str,str,int,str,str]

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