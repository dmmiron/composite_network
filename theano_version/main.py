import numpy as np
import sys


import theano


class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(value=np.zeros((n_in, n_out), dtype=theano.config.floatX), name='W')
        self.b = theano.shared(value=np.zeros((n_out,), dtype=theano.config.floatX), name='b')
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred=T.argmax(self.p_y_given_x, axis=1)
    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.ashape[0]), y])
    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                    ('y', target.type, 'y_pred', self.y_pred.type))
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden layer
                                                                                                                                                                      """
         self.input = input

         W_values = np.asarray(rng.uniform(
                 low=-np.sqrt(6. / (n_in + n_out)),
                 high=np.sqrt(6. / (n_in + n_out)),
                 size=(n_in, n_out)), dtype=theano.config.floatX)
         if activation == theano.tensor.nnet.sigmoid:
             W_values *= 4
         
         self.W = theano.shared(value=W_values, name='W')
         b_values = np.zeros((n_out,), dtype=theano.config.floatX)
         self.b = theano.shared(value=b_values, name='b')

         self.output = activation(T.dot(input, self.W) + self.b)
         sef.params = [self.W, self.b]

class MLP(object):
    """Multi-Laer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the sigmoid function (defined here by a ''HiddenLayer'' class) while the top layer is a softmax layer (defined here by a ''Logistic Regression'' class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out):
        self.hiddenLayer = HiddenLayer(rng = rng, input = input, n_in = n_in, n_out = n_hidden, activation = T.tanh)
        self.logRegressionLayer = LogisticRegression(input=self.hiddenLayer.output, n_in=n_hidden, n_out=n_out)

        self.L1 = abs(self.hiddenLayer.W).sum() + abs(self.logRegressionLayer.W).sum()
        self.L2_sqr = (self.hiddenLayer.W ** 2).sum() + self.logRegressionLayer.W ** 2).sum()

        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        self.errors = self.logRegressionLayer.errors
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

        cost = classifier.negative_log_likelihood(y) + L1_reg * classifier.L1 + L2_reg * classifier.L2_sqr

        gparams = []
        for param in classifier.params:
            gparam = T.grad(cost, param)
            gparams.append(gparam)

        updates = []
        for param, gparam in zip(classifier.params, gparams):
            updates.append((param, param - learning_rate * gparam))

        train_model = theano.function(inputs=[index], outputs=cost, updates=updates,
                givens = {
                    x: train_set_x[index * batchsize:(index + 1) * batch_size],
                    y: train_set_y[index * batchsize:(index + 1) * batch_size]})

