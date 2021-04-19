#Implementing regularized Non-negative Matrix factorization using Regularized gradient descent
import sys, numpy as np
import os
from numpy import genfromtxt
import codecs
from numpy import linalg as LA
import pymp
import eval
import time
from datetime import timedelta

#build movie dicitionary with line no as numpy movie id ,its actual movie id as the key.
def build_movies_dict(movies_file):
    start = time.time()
    i = 0
    movie_id_dict = {}
    with codecs.open(movies_file, 'r', 'latin-1') as f:
        for line in f:
            if i == 0:
                i = i+1
            else:
                movieId,title,genres = line.split(',')
                movie_id_dict[int(movieId)] = i-1
                i = i +1
    CostTime = round(time.time() - start, 3)
    # print('[build_movies_dict]: ',str(timedelta(seconds=CostTime)))
    return movie_id_dict

#Each line of i/p file represents one tag applied to one movie by one user,
#and has the following format: userId,movieId,tag,timestamp
#make sure you know the number of users and items for your dataset
#return the sparse matrix as a numpy array
def read_data(input_file,movies_dict):
    start = time.time()
    #no of users
    users = 718
    #no of movies
    movies = 8927
    # X = np.zeros(shape=(users,movies))
    X = np.full((users,movies),np.nan)
    i = 0
    with open(input_file,'r') as f:
        for line in f:
            if i == 0:
                i = i +1
            else:
                #print "i is",i
                user,movie_id,rating,timestamp = line.split(',')
                #get the movie id for the numpy array consrtruction
                id = movies_dict[int(movie_id)]
                #print "user movie rating",user, movie, rating, i
                X[int(user)-1,id] = float(rating)
                i = i+1
    CostTime = round(time.time() - start, 3)
    # print('[read_data]: ',str(timedelta(seconds=CostTime)))
    return X

#non negative regulaized matrix factorization implemention
def matrix_factorization(X,P,Q,K,steps,alpha,beta):
    Q = Q.T
    r = []
    for step in range(steps):
        #update P
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if X[i][j] > 0 :
                    eij = X[i][j] - np.dot(P[i,:],Q[:,j]) #(r-PQ)
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * ( 2 * eij * Q[k][j] - (beta * P[i][k]))

        #update Q
        for j in range(X.shape[1]):
            for i in range(X.shape[0]):
                if X[i][j] > 0 :
                    eij = X[i][j] - np.dot(P[i,:],Q[:,j]) #(r-PQ)
                    for k in range(K):
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - (beta * Q[k][j]))

        #compute total error
        error = 0
        #for each user
        for i in range(X.shape[0]):
            #for each item
            for j in range(X.shape[1]):
                if X[i][j] > 0:
                    error += np.power(X[i][j] - np.dot(P[i,:],Q[:,j]),2)
        r.append(error)
        if error < 0.001:
            break
    return P, Q.T

def np2pymp(x, thr):
    result = pymp.shared.array((x.shape[0], x.shape[1]))
    with pymp.Parallel(thr) as p:
        for i in p.range(0,x.shape[0]):
            for j in range(x.shape[1]):
                result[i][j] = x[i][j]
    return result

#main function
def main(X,K,stepnum):
    N= X.shape[0] #no of users
    M = X.shape[1] #no of movies
    P = np.random.rand(N,K)
    #P: an initial matrix of dimension N x K, where is n is no of users and k is hidden latent features
    Q = np.random.rand(M,K)
    #Q : an initial matrix of dimension M x K, where M is no of movies and K is hidden latent features
    #steps : the maximum number of steps to perform the optimisation, hardcoding the values
    #alpha : the learning rate, hardcoding the values
    #beta  : the regularization parameter, hardcoding the values
    steps = stepnum
    alpha = 0.0002
    beta = float(0.02)
    start = time.time()
    estimated_P, estimated_Q = matrix_factorization(X,P,Q,K,steps,alpha,beta) 
    CostTime = round(time.time() - start, 3)
    print('[matrix_factorization] Using {} steps: '.format(stepnum),str(timedelta(seconds=CostTime)))
    filename = 'time_mf_serial.txt'
    if filename not in os.listdir():
        f = open(filename,'w')
    with open(filename, 'a') as the_file:
        the_file.write(str(stepnum) + ',' + str(CostTime)+'\n')
    #Predicted numpy array of users and movie ratings
    modeled_X = np.dot(estimated_P,estimated_Q.T)
    np.savetxt('mf_result.txt', modeled_X, delimiter=',')
    return modeled_X

if __name__ == '__main__':
    #mf.py <rating file> <no of hidden features> <movie mapping file>
    if len(sys.argv) == 5:
        ratings_file =  sys.argv[1]
        no_of_features = int(sys.argv[2])
        movies_mapping_file = sys.argv[3]
        stepnum = int(sys.argv[4])
        #build a dictionary of movie id mapping with counter of no of movies
        movies_dict = build_movies_dict(movies_mapping_file)
        #read data and return a numpy array
        numpy_arr = read_data(ratings_file,movies_dict)
        modeled_X = main(numpy_arr,no_of_features, stepnum) #comment out if using train and test below
        '''
        #train&test
        train, test = eval.split_train_test(numpy_arr)
        #main function
        modeled_X = main(train,no_of_features, thr)
        rmse = eval.rmse(test, modeled_X)
        #print (rmse)
        '''