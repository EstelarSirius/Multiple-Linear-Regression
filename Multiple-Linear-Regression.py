import numpy as np
import matplotlib.pyplot as plt
from time import time

def load_data(filename):
  X = np.loadtxt(filename, delimiter=';', skiprows=1)
  Y = X[:, X.shape[1]-1] 
  print(X)
  Y = np.expand_dims(Y, axis=1)
  X = X[:, 0:X.shape[1]-1]
  return X, Y

def calculate_h_theta(X, thetas, m):
  h_theta = np.dot(X,thetas)
  return h_theta
  
def calculate_J(X, Y, thetas):
  J = 0
  m, _ = np.shape(X)
  h_theta = calculate_h_theta(X, thetas, m)
  e = h_theta-Y
  J = (np.dot(np.transpose(e),e))/(2*m)
  return J,thetas

def do_train(X, Y, thetas, alpha, iterations):
  J = np.zeros(iterations)
  m, n = np.shape(X)
  for i in range(iterations):
    J[i], h_theta = calculate_J(X, Y, thetas)
    h_theta = calculate_h_theta(X, thetas, m)
    e = h_theta - Y
    thetas = thetas - ((alpha/m)*(np.dot(np.transpose(X),e)))  
  return J, h_theta

def feature_scaling(X):
  m,n = np.shape(X)
  mean_x = np.mean(X, axis=0)
  standard_deviation = np.std(X, axis=0)
  normalized_X = np.divide(X - mean_x, standard_deviation)
  return normalized_X

if __name__ == "__main__":
  X, Y = load_data('winequality-red.csv')
  X = feature_scaling(X)
  X = np.insert(X, 0, values=1, axis=1) 
  m, n = X.shape
  thetas = np.zeros((n,1))
  alpha = 0.04
  inicio = time()
  J, h_theta = do_train(X, Y, thetas, alpha=alpha, iterations=2000)
  fim = time()

  print("Tempo de execução: {}".format(fim-inicio))

  plt.figure(1)
  plt.plot(J)
  plt.title(r'$J(\theta$) vs iterações')
  plt.ylabel(r'$J(\theta$)', rotation=0)
  plt.xlabel("iteração")
  plt.show()
