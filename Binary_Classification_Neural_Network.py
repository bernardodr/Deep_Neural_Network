#Neural Network with one hidden layer to classify lables in a non linearly separable dataset (Binary classfication)
#functions in google colab

import numpy as np
import matplotlib.pyplot as plt
import keras
from sklearn import datasets
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

np.random.seed(0)

#data 2 circles set
n_pts = 500
X, y = datasets.make_circles(n_samples=n_pts, random_state = 123, noise=0.1, factor=0.2)
plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])


#NN framework
model = Sequential()
model.add(Dense(4, input_shape=(2,), activation ='sigmoid')) #hidden layer
model.add(Dense(1, activation='sigmoid')) #output layer
model.compile(Adam(lr = 0.01), 'binary_crossentropy', metrics=['accuracy']) #learning rate, error/loss funtion, accuracy tool


#model training
h = model.fit(x=X, y=y, verbose= 1, batch_size = 20, epochs = 100, shuffle = 'true') #training data, which lable into class, verbosity: progress of epochs, batch of epochs, number of epochs, shuffle DS: ensuring not stuck in local miniumm


#accuracy plot
plt.plot(h.history['accuracy'])
plt.xlabel('epoch')
plt.legend('accuracy')
plt.title('accuracy')


#Loss plot
plt.plot(h.history['loss'])
plt.xlabel('epoch')
plt.legend('loss')
plt.title('loss')


# with the trained model, correlated each coordinate with prediction prob 'p', plotting contours, representing distinc prob levels. TF ploting classifaction boundary i.e decision Boundary with DS
def plot_decision_boundary(X, y, model):
  x_span = np.linspace(min(X[:, 0]) - 0.25, max(X[:, 0])+ 0.25)
  y_span = np.linspace(min(X[:, 1]) - 0.25, max(X[:, 1])+ 0.25)
  xx, yy = np.meshgrid(x_span, y_span)
  xx_, yy_ = xx.ravel(), yy.ravel()
  grid = np.c_[xx_, yy_]
  pred_func = model.predict(grid)
  z = pred_func.reshape(xx.shape)
  plt.contourf(xx, yy, z)


plot_decision_boundary(X, y, model)
plt.scatter(X[:n_pts,0], X[:n_pts,1])
plt.scatter(X[n_pts:,0], X[n_pts:,1])
#inside label of 1 outside label of 0
#dark = lowest prob level light= highest prob level


#I can now feed the model x and y values and it will output a prediction
plot_decision_boundary(X, y, model)
plt.scatter(X[:n_pts,0], X[:n_pts,1])
plt.scatter(X[n_pts:,0], X[n_pts:,1])
x = 0.1
y = 0
point = np.array([[x,y]])
prediction = model.predict(point)
plt.plot([x], [y], marker="o", markersize=10, color="red")
print("prediction is:", prediction)
#visual rep of classfication
