import numpy as np 
import matplotlib.pyplot as plt 
import scipy as sp 
import csv
import math
from datetime import datetime,date,timedelta
# NN imports
import pandas
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# fix random seed for reproducibility
np.random.seed(27)


# Smash Ultimate Blog Predictor
def main():
	# import data
	labels,data = readin()

	for fighter in data:
		fighter[2] = workdays(fighter[2],date(2018,6,12))*1.0
	labels[2] = 'Weekdays since E3'
	print(labels)
	print(data)

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