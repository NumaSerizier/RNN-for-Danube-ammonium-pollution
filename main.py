import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K

# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset
col_list = [ "Ammonium"]


dateparse = lambda dates: pd.datetime.strptime(dates, '%d/%m/%Y')
col_list = ["Ammonium"]
dataframe = read_csv('Danube ammonium level Time Series.csv')#, usecols=col_list,sep=';', engine='python')
print(dataframe)
dataset = dataframe.values
dataset = dataset.astype('float32')

plt.figure(figsize=(14,8))
plt.plot(dataset)
plt.show()

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = 252
test_size = 12
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
       
# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)  

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(50, input_shape=(1, look_back)))
model.add(Dense(1))

# Compile model
epochs = 100
learning_rate = 0.1
decay_rate = learning_rate / epochs

#optimizer = adam_v2.Adam
model.compile(loss=root_mean_squared_error, optimizer='rmsprop')
model.fit(trainX, trainY, epochs=epochs, batch_size=1, verbose=2)

	

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)



# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

print('RMSE: '+str(math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))))
print('MSE: '+str(mean_squared_error(testY[0], testPredict[:,0])))
print('MAE: '+str(mean_absolute_error(testY[0], testPredict[:,0])))
print('RMSE: '+str(math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))))






# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

# plot baseline and predictions
plt.figure(figsize=(14,8))
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
