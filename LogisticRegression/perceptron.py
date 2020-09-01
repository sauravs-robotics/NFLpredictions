import numpy as np

#sigmoid acitivation function
def sigmoid(x):
  s = 1/(1+np.exp(-x))
  return s

#forward propogation of Neural network
def propagate(w, b, X_train, X_validation, Y_train, Y_validation, regularization, lamd = 0):

  mt = X_train.shape[1]
  mv = X_validation.shape[1]

  #FORWARD PROP
  #train set
  #calculating loss function using binary cross entropy loss for training data
  A = sigmoid(np.dot(w.T,X_train) + b)
  p1 = np.multiply(Y_train,np.log(A))
  y_not = 1-Y_train
  a_not = 1-A
  p2 = np.multiply(y_not,np.log(a_not))
  # r is used if regularizing to prevent overfitting
  r = np.sum(np.square(w))*lamd / (2*mt)
  J = p1 + p2
  cost_train = -(np.sum(J))/mt
  if regularization:
    cost_train += r

  #val set
  #calculating loss function using binary cross entropy loss for validation data
  Av = sigmoid(np.dot(w.T,X_validation) + b)
  p1 = np.multiply(Y_validation,np.log(Av))
  y_not = 1-Y_validation
  a_not = 1-Av
  p2 = np.multiply(y_not,np.log(a_not))
  r = np.sum(np.square(w))*lamd / (2*mv)
  J = p1 + p2
  cost_val = -(np.sum(J))/mv
  if regularization:
    cost_val += r


  # BACK PROP
  #derivative of loss function simplfies to weights = dw*dz.T; b = sum of dz
  dz = A-Y_train
  dw = np.dot(X_train,dz.T)/mt
  if regularization:
    dw += ((w*lamd)/mt)
  db = np.sum(dz)/mt


  cost_train = np.squeeze(cost_train)
  cost_val = np.squeeze(cost_val)
   #create dictionary to retrive values while using optimize function
  theta = {"dw": dw,
             "db": db}

  return theta, cost_train, cost_val

#this function runs forward prop and backprop for number of epochs specified
def optimize(w, b, Xt, Xv, Yt, Yv, epochs, alpha, print_cost, regularization, lamd=0):

    costs_t = []
    costs_v = []

    for i in range(epochs):


        # Cost and gradient calculation (≈ 1-4 lines of code)
        ### START CODE HERE ###
        g, cost_t, cost_v = propagate(w, b, Xt, Xv, Yt, Yv, regularization, lamd)
        ### END CODE HERE ###

        # Retrieve derivatives from grads
        dw = g["dw"]
        db = g["db"]

        w = w - alpha*dw
        b = b - alpha*db
        #save costs for graph of loss function over time
        costs_t.append(cost_t)
        costs_v.append(cost_v)

        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    #save values of theta if needed for output/analyze perceptron for debugging
    theta = {"w": w,
              "b": b}

    dtheta = {"dw": dw,
             "db": db}

    return theta, dtheta, costs_t, costs_v

#if running on a seperate test set after training model
def make_predictions(X, Y, theta):

    w = theta["w"]
    b = theta["b"]
    # Predict test/train set examples
    Z = np.dot(w.T,X)+b
    A = sigmoid(Z)
    Yhat = (A > 0.5).astype(int)


    # Print error
    test_acc = Yhat == Y.astype(int)
    print("train accuracy: {} %".format(100*(np.sum(train_acc)/Y.size)))

#combine all earlier functions for full perceptron model implementation
def model(w,b, X_train, Y_train, X_validation, Y_validation, num_iterations, learning_rate, print_cost, regularization, lamd = 0):


    theta, dtheta , costs_t, costs_v = optimize(w, b, X_train, X_validation, Y_train, Y_validation, num_iterations, learning_rate, print_cost, regularization, lamd)

    w = theta["w"]
    b = theta["b"]
    Z_train = np.dot(w.T,X_train)+b
    A_train = sigmoid(Z_train)
    Z_val = np.dot(w.T,X_validation)+b
    A_val = sigmoid(Z_val)
    Yhat_train = (A_train > 0.4).astype(int)
    Yhat_validation = (A_val > 0.4).astype(int)


    # Print train/test Errors
    train_acc = Yhat_train == Y_train.astype(int)
    val_acc = Yhat_validation == Y_validation.astype(int)
    print("train accuracy: {} %".format(100*(np.sum(train_acc)/Y_train.size)))
    print("validation accuracy: {} %".format(100*(np.sum(val_acc)/Y_validation.size)))





    d = {"train_costs": costs_t,
         "val_costs": costs_v,
         "Yhat_test": Yhat_validation,
         "Yhat_train" : Yhat_train,
         "w" : w,
         "b" : b}

    return d


#enter number of seasons to test data on
nuseasons = int(input("Please enter number of seasons to examine:\n"))
endyear = 2019
startyear = 2019- nuseasons + 1
years = np.linspace(startyear, endyear, num=nuseasons, dtype = int)
for i in range(0,nuseasons):
  y = years[i]
  ranks = "1orderedrankings" + str(y) + ".csv"
  results = "1orderedresults" + str(y) + ".csv"
  if i>0:
    Xnew = np.loadtxt(open(ranks, "rb"), delimiter=",", skiprows=0)
    Ynew = np.loadtxt(open(results, "rb"), delimiter=",", skiprows=0)
    X = np.vstack((X,Xnew))
    Y = np.hstack((Y, Ynew))
  else:
    X = np.loadtxt(open(ranks, "rb"), delimiter=",", skiprows=0)
    Y = np.loadtxt(open(results, "rb"), delimiter=",", skiprows=0)


#normalize data to zero mean, 1 standard deviation; this helps improve gradient
#descent speed since if inputs were different values optimization may take longer
def normalize(X):
  col = X.shape[1] - 1
  me = np.mean(X, axis = 0)
  sd = np.std(X, axis = 0)
  for i in range(0,col):
    X[:,i] = (X[:,i] - me[i])/sd[i]
  return X

#Preprocessing input data is very important!
# Step 1: Shuffle (X, Y) and normalize
m = X.shape[0]

#shuffles data properly for more randomized values to prevent overfitting;
p4 = list(np.random.permutation(m))
#making sure X and Y are shuffled in same order
X = X[p4, :]
Y = Y[p4]

sz = X.shape

#test/validation split for each of the combinations of seasons
#change to split validation/training data differently
split = 0.75 #75% of data points go to training, rest for validation
trainsplit = int(np.floor(sz[0]*split))

#reshape data to align matrix multiplication with neural network (must be m*n, m=training examples, n= number of inputs)
X2_train = X2[0:trainsplit,:]
X2_train = X2_train.T
Y2_train = Y2[0:trainsplit]
X2_test = X2[trainsplit:sz[0],:]
X2_test = X2_test.T
Y2_test = Y2[trainsplit:sz[0]]

#print shapes for further understanding of perceptron if needed
#print(X4_train.shape)
#print(X4_test.shape)




#RUNNING PERCEPTRON NETWORK AND GRAPHING RESULTS
weights = np.matrix([np.zeros(inputs[1])]).T #initialize weights
b = 0.
iterations = 4000
showcost = False
alpha = 0.01
print("1 season:")
modelhist1 = model(weights,b, X1_train, Y1_train, X1_test, Y1_test, iterations, alpha, showcost, False)
print("2 season:")
cost1 = modelhist1["train_costs"]
vcost1 = modelhist1["val_costs"]
plt.plot(cost1, 'g')
plt.plot(vcost1,'g--')
plt.title('model cost')
plt.ylabel('cost')
plt.xlabel('epoch')
plt.show()
