import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

month = ['December','March','June','September','December','January']
year = ['2019','2020','2020','2020','2020','2021']
m=0
y=0
x=[]

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

np.random.seed(7)

def preprocessing(data):
	scaler = MinMaxScaler(feature_range=(0, 1))
	data = scaler.fit_transform(data)
	return scaler, data

def spliting(size, look_back, data):
	train_size = int(len(data) * size)
	train, test = data[0:train_size,:], data[train_size:len(data)-7,:]
	trainX, trainY = create_dataset(train, look_back)
	testX, testY = create_dataset(test, look_back)
	trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
	testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
	return trainX, trainY, testX, testY, data[len(data)-7:len(data),:]

def training(trainX, trainY, epoch, batch):
	model = Sequential()
	model.add(LSTM(units=64,return_sequences=True))
	model.add(LSTM(units=32,return_sequences=False))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=["accuracy"])
	model.fit(trainX, trainY, epochs=epoch, batch_size=batch, verbose=1)
	return model

def fitting(trainX, trainY, epoch, batch, model):
	model.fit(trainX, trainY, epochs=epoch, batch_size=batch, verbose=1)
	return model

def predicion(model, trainX, testX, trainY, testY, scaler, new_data):
	trainPredict = model.predict(trainX)
	testPredict = model.predict(testX)
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY = scaler.inverse_transform([trainY])
	testPredict = scaler.inverse_transform(testPredict)
	testY = scaler.inverse_transform([testY])
	new_data = scaler.inverse_transform(new_data)
	return testY, testPredict, trainPredict, new_data

def RMSE(testY, testPredict):
	testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
	print('Test Score: %.2f RMSE' % (testScore))
	return testScore

def ploting(x, testPredict,data,output,month,year):
	for i in range(0,len(data)):
		x.append(i)
		i+=1
	plt.figure(figsize=(12,5))
	plt.xticks(np.arange(0,(len(data)+int(len(data)/15)),int(len(data)/15)))
	plt.grid(color='b', ls = '-.', lw = 0.25)
	plt.gca().set_facecolor((1.0, 0.992, .816))
	plt.plot(x,data, 'b')
	plt.plot(x[-len(testPredict)-7:-7], testPredict, 'r')
	plt.plot(x[-7:], output,'y')
	plt.ylabel("Relative Humidity")
	plt.xlabel("No of Samples")
	plt.title("Relative Humidity \n Duration: January-2016 to " + month + "-" + year + "  \n Algorithm:LSTM")
	del x[:]
	return plt

def insampleprediction(model,new,scale):
  new = new.reshape(1,-1)
  temp = list(new)
  temp = temp[0].tolist()
  lst_output = []
  nos = 7
  i=0
  while(i<nos):
    if(len(temp)>nos):
      new=np.array(temp[1:])
      new = new.reshape(1,-1)
      new=new.reshape((1,7,1))
      op = model.predict(new, verbose=1)
      temp.extend(op[0].tolist())
      temp = temp[1:]
      lst_output.extend(op.tolist())
      i = i+1
    else:
      new=new.reshape((1,7,1))
      op = model.predict(new, verbose=1)
      temp.extend(op[0].tolist())
      lst_output.extend(op.tolist())
      i=i+1
  return(scale.inverse_transform(lst_output))

dataframe = read_csv('file name with path', usecols=["column number"])
dataset = dataframe.values
dataset = dataset[:len(dataset)].astype('float64')
print(len(dataset))

look = 7
epoch = 1
batch = 100

for i in range(4*365,int(len(dataset))):
	scale, new_data = preprocessing(dataset[:i]) 
	trainx,trainy, testx, testy, new = spliting(0.8,look,new_data)
	mdel = training(trainx, trainy, epoch, batch)
	testy, testpredict, trainpredict, data_new = predicion(mdel,trainx,testx,trainy,testy,scale,new_data)
	output=insampleprediction(mdel,new,scale)
	print(testpredict)
	score = RMSE(testy,testpredict)
	plt = ploting(x,testpredict,data_new,output,month[m],year[y])
	plt.show()
	while(i<int(len(dataset))):
		m+=1
		y+=1
		scale,new_data = preprocessing(dataset[:i+91])
		trainx,trainy, testx, testy, new = spliting(0.8,look,new_data) 
		mdel = fitting(trainx, trainy, epoch, batch, mdel)
		testy, testpredict, trainpredict, data_new = predicion(mdel,trainx,testx,trainy,testy,scale,new_data)
		output=insampleprediction(mdel,new,scale)
		score = RMSE(testy,testpredict)
		plt = ploting(x,testpredict,data_new,output,month[m],year[y])
		plt.show()
		i += 91  
	break
#mdel.save("incremental_LSTM_temp")
