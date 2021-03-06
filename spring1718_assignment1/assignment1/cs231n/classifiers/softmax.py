import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # f = W*x ---these are the weights, or the scores
  scores = W.T.dot(X.T).T # 10x3073 x 3073x500 => 500x10
  scores_shifted = scores - np.max(scores)

  #W = 3073x10
  #X = 500*3073
  for i in xrange(num_train):
    p = np.exp(scores_shifted[i]) / np.sum(np.exp(scores_shifted[i]))
    loss += - math.log(p[y[i]])
    for j in xrange(num_classes):
      if j == y[i]:
        dW[:, j] += (p[j]-1)*X[i].T #shape
      else:
        dW[:,j] += p[j]*X[i].T

  loss /= num_train
  loss += reg*.5*np.sum(W*W)

  dW /= num_train
  dW += reg*W #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores  = W.T.dot(X.T).T
  scores_shifted = scores - np.max(scores)

  normalized_scores = (np.exp(scores_shifted.T) / np.sum(np.exp(scores_shifted), axis =1)).T
  loss += - np.sum(np.log(normalized_scores[range(num_train), y]))

  grad_scores = np.where(range(10) == y, normalized_scores-1, normalized_scores)

  loss /= num_train
  loss += reg*.5*np.sum(W*W)

  dW += (grad_scores.T.dot(X)).T
  dW /= num_train
  dW += reg*W
 
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

