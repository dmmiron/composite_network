This repository contains two networks specified in .yaml files. The berlin
networks are fully connected networks with 3 hidden layers of 500 units each.
The composite networks have the same structure, but at each hidden layer the
original input is concatenated to the output before being sent to the next
layer.

classify.py provides both a pylearn2 implementation for classification and a
slow numpy implementation for testing intermediate results.

driver.py can be used to run multiple different versions of the two networks
to generate multiple data points for speed and performance testing.

plot.py can be used to plot the accuracy of the networks versus training time
and versus training epochs. 

fixed_autoencoder.pkl contains a premade autoencoder layer with the identity
matrix. This is read by the composite_network.yaml and should be copied with
the composite_network.yaml.


