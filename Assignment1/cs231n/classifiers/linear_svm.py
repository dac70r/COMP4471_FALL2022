from ast import Index
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights. 3073 * 10
  - X: A numpy array of shape (N, D) containing a minibatch of data. 500 * 3073
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means 500 * 1 
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """ 
  dW = np.zeros(W.shape) # initialize the gradient as zero 3073 * 10

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0] 
  loss = 0.0
  for i in xrange(num_train): # in range of (500)
    scores = X[i].dot(W) # shape is (10,1) 1 * 3073 x 3073 * 10 itr 500 times
    #print(scores.shape)
    correct_class_score = scores[y[i]] # selects the score that is correct
    for j in xrange(num_classes): #in range of (10)
      if j == y[i]:
        continue # calculate the loss for all except y == i 
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:  
        loss += margin
        dW[:,j] += X[i]
        dW[:, y[i]] += -X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg * W * 2

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_train = X.shape[0]
  scores = X @ W #(500 * 10)
  correct_class_score = scores[np.arange(num_train),y][:, np.newaxis] # all rows in scores, extract the correct score (500, )
  #print(y.shape) # (500,)
  #print(scores.shape) # (500,10)
  #print(correct_class_score.shape) (500,1)
  margin = scores - correct_class_score + 1 # subtraction by broadcasting
  #print(margin) #(500,1) 
  margin[np.arange(num_train),y]=0 # clearing values y == i
  #print(margin)
  loss = loss + margin[margin>0].sum()/num_train # sum over all values that are bigger than 1
  loss = loss + reg * np.sum(W*W) 
   
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  counts = (margin>0).astype(int) 
  #print(counts)
  counts[range(num_train),y] = -np.sum(counts,axis=1)
  #print(counts) # (500,10)
  dW = dW + np.dot(X.T, counts)/num_train  #(3073,500)*(500,10)
  dW = dW + reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
