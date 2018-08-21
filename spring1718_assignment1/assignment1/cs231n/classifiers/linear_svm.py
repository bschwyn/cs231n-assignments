import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

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
  scores = X.dot(W) #500 x 10
  y_tiled = np.tile(y,(10,1)).T # 3,3,3,3,3,3
  ccs = np.where(range(10)==y_tiled,scores,0) #[0,0, 0 , score , 0, 0 ]
  ccs = np.sum(ccs, axis = 1) #collasping to vector of just the scores at yi
  correct_class_scores = np.tile(ccs,(10,1)).T #net result of tiling matrix w/ scores at yi
  delta = np.ones(scores.shape)
  margins = scores - correct_class_scores + delta #500 x 10
  #loss is sum of all values in margins, except where j== y[i]
  #set those values to 0. This matrix stored as "L"

  L = np.where(range(10) == y_tiled, 0,margins)
  loss_matrix = np.maximum(L, np.zeros(L.shape))
  loss += np.sum(loss_matrix)
  loss /= X.shape[0]

  #indicator = margins > 0
  #count = np.sum(np.where(range(10)==y_tiled, 0, indicator), axis = 1)
  indicator =  L > 0 # this is 1 if score - ccs + 1 > 0 for each spot, with a zero at j==yi

  count_vector = np.sum(indicator,axis = 1).reshape(-1,1) #
  count_matrix = np.where(range(10)==y_tiled, count_vector,0)
  dW += (indicator.T.dot(X)).T

  dW += -(count_matrix.T.dot(X)).T
  dW /= X.shape[0]

  dW += reg*W

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
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
