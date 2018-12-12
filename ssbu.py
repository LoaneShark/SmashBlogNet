import numpy as np 
import matplotlib.pyplot as plt 
import scipy as sp 
import argparse
import csv
import math
from datetime import datetime,date,timedelta
import calendar
import warnings
# NN imports
import pandas
from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed, Embedding
import keras.activations as ka
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.exceptions import NotFittedError

#python ssbu.py -e 350 -l 5 -d -n 10 -r -o

parser = argparse.ArgumentParser()
parser.add_argument('-v','--verbosity',type=int,default=0)
parser.add_argument('-b','--batch_size',type=int,default=1)
parser.add_argument('-e','--epochs',type=int,default=250)
parser.add_argument('-l','--lag',type=int,default=1)
parser.add_argument('-s','--seed',type=int,default=-1)
parser.add_argument('-n','--n_runs',type=int,default=1)
parser.add_argument('-f','--fighter_post',action='store_true')
parser.add_argument('-d','--day_of_week',action='store_true')
parser.add_argument('-p','--plot',action='store_true')
parser.add_argument('-u','--unsupervised',action='store_true')
parser.add_argument('-c','--calc_error',action='store_true')
parser.add_argument('-r','--retro',action='store_true')
parser.add_argument('-R','--travel_back',type=int,default=0)
parser.add_argument('-m','--missing_days',type=int,default=0)
parser.add_argument('-a','--activation',type=str,default='sigmoid')
parser.add_argument('-t','--third_party',action='store_true')
parser.add_argument('-o','--company',action='store_true')
parser.add_argument('-A','--plot_acc',action='store_true')
parser.add_argument('-T','--show_tail',action='store_true')
args = parser.parse_args()
#print(args.verbosity)

# fix random seed for reproducibility
if args.seed >= 0:
	#np.random.seed(27)
	np.random.seed(args.seed)

verbosity = args.verbosity
fighter_post = args.fighter_post
day_of_week = args.day_of_week
is_supervised = not(args.unsupervised)
consolidate_retros = args.retro
blank_days = args.missing_days
group_by_company = args.company
consolidate_third_parties = args.third_party
plot_accuracy = args.plot_acc
peek_tail = args.show_tail

# Smash Ultimate Blog Predictor
def main():
	# import data
	labels,fighters,others = readin()
	if day_of_week:
		labels = np.append(labels,["Day of the Week"])
	if verbosity >= 1:
		print("Analyzing %d fighters..." %(len(fighters)))

	# clean up data and add in missing info
	last = 0
	blankday = np.array(["None",-1,None,-1,"None","None","None","None","None",-1,"None","None"])

	if day_of_week:
		cleandata = np.zeros((1,fighters.shape[1]+1),dtype='object')
		cleandata[0] = np.append(blankday,["None"])
	else:
		cleandata = np.zeros((1,fighters.shape[1]),dtype='object')
		cleandata[0] = np.copy(blankday)
	
	# calculate weekday count for each post
	# add in missing days with blankday placeholders
	today = date.today()
	for fighter in fighters:
		if day_of_week:
			day_i = fighter[2].weekday()
			fighter = np.append(fighter,[calendar.day_name[day_i]])
		fighter[2] = workdays(fighter[2],date(2018,6,12))*1.0
		diff = fighter[2] - last
		while diff >= 2:
			diff = diff-1
			tempday = np.copy(blankday)
			tempday[2] = fighter[2] - diff
			if day_of_week:
				tempday = np.append(tempday,[calendar.day_name[int((day_i-diff)%7)]])
			#print(tempday.shape)
			cleandata = np.append(cleandata,[tempday],axis=0)
		cleandata = np.append(cleandata,[fighter],axis=0)
		last = fighter[2]
	# pad missing days since last blog post 
	last = cleandata[-1,2]
	if blank_days == -1:
		padcount = int(workdays(today)-last)
	else:
		padcount = blank_days+1

	for i in range(0,padcount):
		tempday = np.copy(blankday)
		tempday[2] = last+i+1
		if day_of_week:
			tempday = np.append(tempday,[calendar.day_name[(day_i+i)%7]])
		cleandata = np.append(cleandata,[tempday],axis=0)

	# calculate weekday count for each non-fighter post
	if day_of_week:
		others = np.array([np.append(item,[calendar.day_name[item[2].weekday()]]) for item in others])
	others = np.array([np.append(item[:2],np.append([workdays(item[2],date(2018,6,12))*1.0],item[3:])) for item in others])
	#print(others)
	for idx in range(len(cleandata)):
		if cleandata[idx,0] == "None":
			#print(line)
			temp = np.array([item for item in others if item[2] == cleandata[idx,2]])
			if temp.size>0:
				#print(temp)
				cleandata[idx] = np.array(temp[0])

	# remove specified number of days
	if args.travel_back > 0:
		cleandata = cleandata[:-args.travel_back]

	data = cleandata[1:]
	if verbosity >= 3:
		print(data.shape,data)
	elif peek_tail:
		print("Last 5 elements of dataset...")
		print(data[-5:])

	labels[2] = 'Weekdays since'
	# establish mapping for labels to their index
	labelkeys = {}
	for i in range(len(labels)):
		labelkeys[labels[i]] = i
	
	# select just the data columns we want, and reorder them
	if day_of_week:
		titles = ['Number','Type','Series','Day of the Week','Weekdays since','Game Added','Game Count','3rd Party?']
	else:
		titles = ['Number','Type','Series','Weekdays since','Game Added','Game Count','3rd Party?']
	features = np.array([data[:,labelkeys[title]] for title in titles],dtype='object')
	features = np.array([features[:,i] for i in range(len(data))])
	
	if verbosity >= 4:
		print(titles,"\n",features)

	labels = titles
	#if verbosity >= 3:
	#	print(labels)
	#	print(data)
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
	recent = scaled[-(args.lag):]
	#print(recent)

	# frame as supervised learning
	dataset = series_to_supervised(scaled,args.lag,1)

	if args.lag < 1:
		# drop columns that don't matter
		if (fighter_post):
			# drop all outputs except fighter number
			dataset.drop(dataset.columns[range(n_features+1,2*n_features)],axis=1, inplace=True)
		else:
			# drop all outputs except post type & series
			dataset.drop(dataset.columns[range(n_features+3,2*n_features)],axis=1, inplace=True)
			dataset.drop(dataset.columns[n_features],axis=1, inplace=True)

	if verbosity >= 1:
		print(dataset.head())

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
	#trainX, trainY = create_dataset(train, args.lag)
	#testX, testY = create_dataset(test, args.lag)

	n_obs = args.lag * n_features
	trainX, testX = train[:,:n_obs], test[:,:n_obs]
	if fighter_post:
		trainY, testY = train[:,-n_features], test[:,-n_features]
	else:
		trainY, testY = train[:,-n_features+1:-n_features+3], test[:,-n_features+1:-n_features+3]
	# reshape
	trainX = trainX.reshape((trainX.shape[0],args.lag,n_features))
	testX = testX.reshape((testX.shape[0],args.lag,n_features))

	if verbosity >= 1:
		print("trainX: ",trainX.shape)
		if verbosity >= 2: print(trainX)
		print("trainY: ",trainY.shape)
		if verbosity >= 2: print(trainY)
	#inp_n = trainX.shape[2]

	# model LSTM network
	model,hist = model_dataset(trainX,trainY,testX,testY,n_features,look_back=args.lag,plot_results=args.plot)

	# make predictions!
	if args.calc_error:
		result, error = model_predict(model,testX,testY,recent.reshape(1,args.lag,n_features),scaler,encoders,titles,calc_rmse=args.calc_error)
	else:
		result, _ = model_predict(model,testX,testY,recent.reshape(1,args.lag,n_features),scaler,encoders,titles,calc_rmse=args.calc_error)

	return result

# create and train the LSTM network
def model_dataset(trainX,trainY,testX,testY,n_feats,look_back=1,plot_results=True):
	if verbosity >= 1:
		print("Generating model")

	#n_feats = trainX.shape[2]
	hidden = 20
	# generate and train network
	model = Sequential()
	#model.add(LSTM(hidden, input_shape=(look_back,n_feats),return_sequences=True))
	model.add(LSTM(hidden, input_shape=(look_back,n_feats)))
	if fighter_post:
		model.add(Dense(1,activation=args.activation))
	else:
		#model.add(TimeDistributed(Dense(2,activation=args.activation),input_shape=(look_back,n_feats)))
		model.add(Dense(2,activation=args.activation))
	#model.compile(loss='mae',optimizer='adam')
	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['categorical_accuracy'])
	history = model.fit(trainX,trainY,epochs=args.epochs,batch_size=args.batch_size,validation_data=(testX,testY),verbose=min([int(verbosity/2),1]),shuffle=False)

	# plot
	if plot_accuracy:
		keystr = "categorical_accuracy"
	else: keystr = "loss"
	if plot_results:
		plt.plot(history.history[keystr],label='train')
		plt.plot(history.history['val_%s' %keystr], label='test')
		plt.title("%s vs. epoch for train & validation sets" %keystr)
		plt.xlabel("epoch")
		plt.ylabel(keystr)
		plt.legend()
		plt.show()

	return model,history

def model_predict(model,testX,testY,mostrecent,scaler,encoders,titles,calc_rmse=False):
	
	if calc_rmse:
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
	else:
		rmse = -999

	# predict next fighter
	predicted = model.predict(mostrecent,batch_size=1,verbose=1)
	#print(predicted)
	if not fighter_post:
		# pad with dummy values so it can be descaled
		dummy = np.append([0],predicted[0],axis=0)
		pad = [0]*(len(titles)-3)
		dummy = np.append(dummy,pad,axis=0)
		#print(dummy)
		dummy = scaler.inverse_transform([dummy])
		predicted = dummy[0,1:3]
		#print(predicted)

	mostrecent = scaler.inverse_transform(mostrecent[0])
	#print(mostrecent)
	last = np.zeros(len(titles),dtype='object')
	for i in range(len(titles)):
		feature = titles[i]
		encoder = encoders[i]
		#print(i,": ",feature)

		# suppress warnings caused by sklearn bug
		warnings.filterwarnings(action='ignore',category=DeprecationWarning)
		#if not fighter_post and (feature == "Series" or feature == "Type"):
		#	predicted[i] = encoder.inverse_transform(predicted[i].astype(int))
		try:
			last[i] = encoder.inverse_transform(mostrecent[-1,i].astype(int))
		except NotFittedError:
			last[i] = mostrecent[-1,i].astype(int)

	#mostrecent = mostrecent.reshape((len(mostrecent),1))
	if verbosity >= 1:
		print("Last Posted Fighter: \n",titles)
		print(" ",last)

	if fighter_post:
		pred_num = predicted[:,0]*66
		pred_num = pred_num.astype(int)
		
		if verbosity >= 1:
			print("Next Predicted Fighter(s): ",pred_num)
		return pred_num,rmse
	else:
		#print(predicted)
		res = np.array([0,0],dtype='object')
		res[0] = encoders[1].inverse_transform(predicted[0].astype(int))
		try:
			res[1] = encoders[2].inverse_transform(predicted[1].astype(int))
		except ValueError:
			if predicted[1].astype(int) == len(encoders[2].classes_):
				res[1] = "New Series"
			else:
				res[1] = "Error: " + str(predicted[1])
		if verbosity >= 1:
			print("Prediction: ",res)
		return res,rmse

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

	if fighter_post:
		y_dim = 1
	else:
		y_dim = 2

	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back),:]

		dataX.append(a)
		if fighter_post:
			dataY.append(dataset[i+look_back,0])
		else:
			dataY.append(dataset[i+look_back,1:3])
	return np.array(dataX), np.array(dataY).reshape(y_dim,len(dataY))

# reads in specified data csv and returns a
# 	numpy array containing the cleaned data
# 	and another containing the labels
def readin(fighterpath='blogdata.csv',itempath='itemdata.csv'):
	data, items = [], []

	# open file
	with open(fighterpath) as csvfile:
		datareader = csv.reader(csvfile)
		for fighter in datareader:
			data.append(fighter)
	with open(itempath) as itemfile:
		itemreader = csv.reader(itemfile)
		for post in itemreader:
			items.append(post)
	labels = data[0]
	n = len(labels)
	data = data[1:]
	items = items[1:]

	# ready to use data 
	data = parsecsv(data,n)
	other = parsecsv(items,n)
	return np.array(labels),data,other

# cleans CSV and assigns proper data types
# returns numpy array of data vectors, without labels
def parsecsv(data,n):
	cleandata = np.zeros([1,n],dtype='object')
	_date = lambda x: datetime.strptime(x, "%m/%d/%y").date()
	_bool = lambda x: True if x == "True" else False
	#parsers = [str,float,_date,int,str,_bool,_bool,_bool,int,str,str]
	parsers = [str,float,_date,int,str,str,str,str,str,int,str,str]

	# combines low-population subgroups into their heirarchical classifier
	for fighter in data:
		parsed = [parse(inp) for parse,inp in zip(parsers,fighter)]
		# consolidate minor nintendos
		if parsed[-2] in ["Wii","Nintendo DS","Electroplankton","Wii Sports","Find Mii","Miiverse","Tomodachi","Nintendogs","Snipperclips","Steel Diver","Pilotwings"]:
			parsed[-2] = "Other Nintendo"
		# consolidate retros
		if consolidate_retros:
			if parsed[-2] in ["Ice Climber","Gyromite","Joy Mech Fight","Excitebike","Game & Watch","Duck Hunt","Balloon Fight","Wrecking Crew"]:
				parsed[-2] = "Retro"
		# consolidate third parties
		if group_by_company or consolidate_third_parties:
			if parsed[-2] in ["Castlevania", "Metal Gear","Bomberman","Contra","Dance Dance Revolution","Frogger"]:
				parsed[-2] = "Konami"
			if parsed[-2] in ["Pac-Man","Tekken","Digimon","Dig Dug","Tales","Dark Souls","Tetris","Galaga","Klonoa"]:
				parsed[-2] = "Namco Bandai"
			if parsed[-2] in ["Mega Man","Street Fighter","Monster Hunter","Ace Attorney","Okami","Devil May Cry","Resident Evil"]:
				parsed[-2] = "Capcom"
			if parsed[-2] in ["Crash Bandicoot","Spyro"]:
				parsed[-2] = "Activision"
			if parsed[-2] in ["Sonic The Hedgehog","Bayonetta","Virtua Fighter","NiGHTS","Puyo Puyo","Megami Tensei","Shenmue","Persona","Etrian Odyssey","Yakuza","Valkyria Chronicles"]:
				parsed[-2] = "Sega"
			if parsed[-2] in ["Minecraft","Halo","Banjo-Kazooie","Conker"]:
				parsed[-2] = "Microsoft"
			if parsed[-2] in ["Rayman","Assassin's Creed","Prince of Persia"]:
				parsed[-2] = "Ubisoft"
			if parsed[-2] in ["Elder Scrolls","DOOM","Fallout","Wolfenstein"]:
				parsed[-2] = "Bethesda"
			if parsed[-2] in ["Final Fantasy","Super Mario RPG","Bravely Default","Octopath Traveler","Dragon Quest","Kingdom Hearts","Chrono Trigger","The World Ends With You","Nier","Tomb Raider"]:
				parsed[-2] = "Square Enix"
			if parsed[-2] in ["Professor Layton","Yo-Kai Watch","Ni no Kuni"]:
				parsed[-2] = "Level-5"
			if parsed[-2] in ["Overwatch","Diablo","Warcraft","Starcraft","Hearthstone"]:
				parsed[-2] = "Blizzard"
			if parsed[-2] in ["Shovel Knight","Shantae","Undertale","Scribblenauts","Cave Story","Celeste","Azure Striker Gunvolt","Tohou","A Hat in Time","Owlboy","Never Alone","Bit Trip","Spelunky","Binding of Isaac","Rivals of Aether","Slap City","Brawlhalla"]:
				parsed[-2] = "Indie"
			if parsed[-2] in ["Dragon Ball","My Hero Academia","Naruto","One Piece","Bleach","Yu-Gi-Oh","Fullmetal Alchemist","Death Note","JoJo's Bizzare Adventure"]:
				parsed[-2] = "Anime"
		if consolidate_third_parties:
			if parsed[-2] in ["Konami","Namco Bandai","Capcom","Activision","Sega","Microsoft","Ubisoft","Bethesda","Square Enix","Level-5","Blizzard"]:
				parsed[-2] = "3rd Party"
		cleandata = np.append(cleandata,[parsed],axis=0)

	return cleandata[1:]

# returns the number of WEEKDAYS between two dates
def workdays(finaldate, refdate = date(2018,6,12)):
	daygen = (refdate + timedelta(x + 1) for x in range((finaldate-refdate).days))
	return sum(1 for day in daygen if day.weekday()<5)

if __name__ == "__main__":
	results = np.zeros([1,2],dtype='str')
	# aggregate results
	for n in range(args.n_runs):
		if n%10 == 0 and n>0:
			print(n)
		results = np.append(results,[main()],axis=0)
	results = results[1:]

	# count & sort unique results
	res = [tuple(row) for row in results]
	#print(res)
	u, c = np.unique(res,return_counts=True,axis=0)
	N = sum([val for val in c])

	# publish results
	print("||-=-+-=-+-=-+-=-+-=-+-=-+-=-+-=-+-=-")
	print("|| Prediction for %s" %date.strftime("%A, %m/%d/%Y"))
	print("||",args.n_runs," function calls: ")
	for x,y in zip(u,c):
		print("||%.1f%%: %s" %(y*100./N, x))
	print("||-=-+-=-+-=-+-=-+-=-+-=-+-=-+-=-+-=-")