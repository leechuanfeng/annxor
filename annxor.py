# the structure of neural network: 
#    input layer with 2 inputs
#    1 hidden layer with 2 units, tanh()
#    output layer with 1 unit, sigmoid()

import numpy as np
import scipy
from scipy.special import expit
import math
def run():
  X = loadData("XOR.txt")
  W = paraIni()
  #print(X.shape)
  intermRslt = feedforward(X,W)
  #print(intermRslt[2])
  Y = X[:, len(X[0])-1:len(X[0])]
  #print(Y)
  Yhat = intermRslt[2]
  #print(Yhat)
  #print(errCompute(Y, Yhat))
  B=backpropagate(X, W, intermRslt, 0.5)
  #print(B)
  
  numIter = [100, 1000, 5000, 10000]
  alp = [0.01, 0.5]
  for i in range(len(numIter)):
    for j in range(len(alp)):
      R=FFMain("XOR.txt", numIter[i], alp[j])
      np.savetxt('Error(numIter=' + repr(numIter[i]) + ',alp='+ repr(alp[j])+ ')' + '.txt', R[0], fmt="%.8f")
      np.savetxt('Output(numIter=' + repr(numIter[i]) + ',alp='+ repr(alp[j])+ ')' + '.txt', R[1], fmt="%.8f")
      np.savetxt('NewHidden(numIter=' + repr(numIter[i]) + ',alp='+ repr(alp[j])+ ')' + '.txt', R[2][0], fmt="%.8f")
      np.savetxt('NewOutput(numIter=' + repr(numIter[i]) + ',alp='+ repr(alp[j])+ ')' + '.txt', R[2][1], fmt="%.8f")
  #R=FFMain("XOR.txt", 10000, 0.5)
  #print("R",R[1])
  
def loadData(Filename):
  X=[]
  count = 0
  
  text_file = open(Filename, "r")
  lines = text_file.readlines()
  
  for line in lines:
    X.append([])
    words = line.split(' ')
    #print(words)
    # convert value of first attribute into float  
    for word in words:
      #print(word)
      X[count].append(float(word))
    count += 1
  X = np.asarray(X)
  n,m = X.shape # for generality
  X0 = np.ones((n,1))
  X = np.hstack((X0,X))
  return X

def paraIni():
  #code for fixed network and initial values
  # parameters for hidden layer, 3 by 3 
  #wh=np.array([[0.1859,-0.7706,0.6257],[-0.7984,0.5607,0.2109]])
  #wh=np.random.random_sample((2,3))
  wh=np.random.uniform(low = -1.0001, high=1.0001, size=(2,3))
  wh[wh>1.0] = 1.0
  wh[wh<-1.0] = -1.0

  # parameter for output layer 1 by 3
  #wo=np.array([[0.1328,0.5951,0.3433]])
  #wo=np.random.random_sample((1,3))
  wo=np.random.uniform(low = -1.0001, high=1.0001, size=(1,3))
  wo[wo>1.0] = 1.0
  wo[wo<-1.0] = -1.0

  return [wh,wo]
  
def feedforward(X,paras):
  tempX = X[:, 0:len(X[0])-1] #x,y
  tempY = X[:, len(X[0])-1:len(X[0])]
  oh = np.tanh(np.dot(paras[0], tempX.transpose()))
  n,m = oh.shape # for generality
  X0 = np.ones((m,1))
  ino = np.vstack((X0.transpose(),oh))
  oo = expit(np.dot(paras[1],ino))
  return [oh,ino,oo]
  
def errCompute(Y,Yhat):
  #this will not have the output from the original array
  sum = 0
  Yo = Yhat[0]

  for k in range(len(Y)):
    sum += pow((Y[k] - Yo[k]),2)
  #sum all values & find error value
  J = sum / (2 * len(Yo))

  return J

def backpropagate(X,paras,intermRslt,alpha):
  #Initializing
  Y = X[:, len(X[0])-1:len(X[0])]
  tempX = X[:, 0:len(X[0])-1]
  oo = intermRslt[2][0]
  ino = intermRslt[1]
  oh = intermRslt[0]
  wh = paras[0]
  wo = paras[1]

  delta = np.multiply(np.multiply((Y.transpose() - oo), oo), (1.0-oo))
  wo = wo + (alpha * np.dot(delta,ino.transpose()))/4.0
  wop = wo[:, 1:len(wo[0])]
  dot =np.dot(wop.transpose(), delta)

  deltah = np.multiply(dot, (1.0-oh*oh))
  wh = wh + alpha * np.dot(deltah, tempX) / 4.0

  return [wh,wo]
  
def FFMain(filename,numIteration, alpha):
  #data load
  X = loadData(filename)
  #
  W = paraIni()
  
  #number of features
  n = X.shape[1]
  
  #error
  errHistory = np.zeros((numIteration,1))
  
  for i in range(numIteration):
    #feedforward
    intermRslt=feedforward(X,W)
    #Cost function
    errHistory[i,0]=errCompute(X[:,n-1:n],intermRslt[2])
    #backpropagate
    W=backpropagate(X,W,intermRslt,alpha)

  Yhat=np.around(intermRslt[2]) 
  return [errHistory,intermRslt[2],W]