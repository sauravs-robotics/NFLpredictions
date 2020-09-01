import numpy as np
import pandas as pd
from google.colab import files
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

#same method as perceptron.py used to select number of seasons to analyze
nuseasons = int(input("Please enter number of seasons to examine:\n"))
endyear = 2019
startyear = 2019- nuseasons + 1
years = np.linspace(startyear, endyear, num=nuseasons, dtype = int)
for i in range(0,nuseasons):
  y = years[i]
  ranks = str(input_type) + "orderedrankings" + str(y) + ".csv"
  results = str(input_type) + "orderedresults" + str(y) + ".csv"
  if i>0:
    Xnew = np.loadtxt(open(ranks, "rb"), delimiter=",", skiprows=0)
    Ynew = np.loadtxt(open(results, "rb"), delimiter=",", skiprows=0)
    X = np.vstack((X,Xnew))
    Y = np.hstack((Y, Ynew))
  else:
    X = np.loadtxt(open(ranks, "rb"), delimiter=",", skiprows=0)
    Y = np.loadtxt(open(results, "rb"), delimiter=",", skiprows=0)

print(X.shape)
print(Y.shape)



#note:since keras implements shuffling of data, this is not needed to be manually
#implemented
 this time
X1 = X
Y1 = Y
m = X1.shape[0]
n = X1.shape[1]

#initializes network
model = Sequential()
#adds dense layers
model.add(Dense(units = 8, input_shape = (n,), activation = 'relu'))
#batch norm helps prevent overfitting/improve overall performance
model.add(BatchNormalization())
model.add(Dense(units = 8, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(units = 4, activation='relu'))
model.add(BatchNormalization())
#model.add(Dense(units = 2, activation='relu'))
#model.add(BatchNormalization())
model.add(Dense(units = 1, activation='sigmoid'))
ag = tf.optimizers.Adagrad(learning_rate=0.001)
adam = tf.optimizers.Adam(learning_rate= 0.001)
model.compile(optimizer = ag,loss= 'binary_crossentropy', metrics=['binary_accuracy'])
h = model.fit(x=X1, y=Y1, verbose=0, batch_size = m, epochs = 2000, shuffle = 'true', validation_split=0.15)


#plotting accuracy and loss 
plt.plot(h.history['binary_accuracy'])
plt.plot(h.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_1', 'validation_1'], loc='lower left')
plt.show()
plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_1', 'validation_1'], loc='upper right')
plt.show()
