import pylearn2.models.mlp as mlp
import pylearn2.models.autoencoder as auto
import numpy as np
#from pylearn2.space import CompositeSpace, Conv2DSpace, VectorSpace, IndexSpace
from pylearn2.space import VectorSpace

if __name__ == '__main__':
    patchSize = 39
    
    layer = mlp.Linear(dim=patchSize ** 2,
                    layer_name = 'fixed_input',
                    irange = 0.01,
                    use_bias=False)

    mlp_stupid = mlp.MLP(layers = [layer], batch_size=None, input_space=None,
                 nvis=patchSize**2, seed=None, layer_name=None)
    
    W_fixed = np.eye(patchSize ** 2).astype(np.float32)
    layer.set_input_space(VectorSpace(dim=patchSize ** 2))
    layer.set_weights(W_fixed)
    


    autoencoder = auto.Autoencoder(patchSize ** 2, patchSize ** 2, None, None, tied_weights=True)

    autoencoder.weights = W_fixed
