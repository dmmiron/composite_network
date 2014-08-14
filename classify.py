import numpy as np
import sys
import glob
import mahotas

import theano
from pylearn2.utils import serial

def normalize_image_float(original_image, saturation_level=0.005):
    sorted_image = np.sort( np.uint8(original_image).ravel() )
    minval = np.float32( sorted_image[ len(sorted_image) * ( saturation_level / 2 ) ] )
    maxval = np.float32( sorted_image[ len(sorted_image) * ( 1 - saturation_level / 2 ) ] )
    norm_image = np.float32(original_image - minval) * ( 255 / (maxval - minval))
    norm_image[norm_image < 0] = 0
    norm_image[norm_image > 255] = 255
    return norm_image / 255.0

def save_image(image, out_name):
    mahotas.imsave(out_name, np.int8(image))
    print "saved image: {0}".format(out_name)

def load_image(image_name):
    image = np.float32(mahotas.imread(image_name))
    image = normalize_image_float(image)
    return image

def classify_image_pylearn2(image, model, patch_dims, batchsize):
    layers = model.layers
    valid_x = image.shape[0] - patch_dims[0] + 1
    valid_y = image.shape[1] - patch_dims[1] + 1

    pixels = [(x,y) for x in range(valid_x) for y in range(valid_y)]
    patchsize = patch_dims[0]*patch_dims[1]
    nbatches = (len(pixels) + batchsize - 1)/batchsize
    model.set_batch_size(batchsize)
    data = model.get_input_space().make_batch_theano()
    outputs = np.float32(np.zeros((nbatches*batchsize, 2)))
    y = model.fprop(data)
    print model
    for batch in range(nbatches):
        start = batch*batchsize
        values = np.float32(np.zeros((batchsize, patchsize)))
        for pixn, pixel in zip(range(batchsize), pixels[start:start+batchsize]):
            values[pixn, :] = image[pixel[0]:pixel[0]+patch_dims[0], pixel[1]:pixel[1]+patch_dims[1]].ravel()
        print y
        
        classify = theano.function([data], [y], name='classify')
        output = np.array(classify(values))
        print output.shape
        print batch
        
        """         
        for layer in model.layers:
            print layer

            y = layer.layers[0].fprop(data)
            print y, data
            classify = theano.function([data], [y], name='classify')
            output = np.array(classify(values))
            print output, "output"
            values = output[0]
            print values, "values"
        """ 
        outputs[start:start+batchsize, :] = output[0]
    return outputs[:len(pixels), :]

def classify(image_names, model_file_name, output_names):
    model = serial.load(model_file_name)
    outputs = []
    patch_dims = (39, 39)
    batchsize = 10000
    for image_name, output_name in zip(image_names, output_names):
        image = load_image(image_name)
        valid_x = image.shape[0] - patch_dims[0] + 1
        valid_y = image.shape[1] - patch_dims[1] + 1
        output_p = classify_image_pylearn2(image, model, patch_dims, batchsize)
        output = output_p
        print output
        output_np = classify_image_np(image, model, patch_dims)
        print output_np
        print output_np.shape
        #output_np = output[:, 0]
        output = output.reshape(valid_x, valid_y)
        save_image(np.int32(np.round(output*255)), "np_"+output_name)
        output = output[:, 0]
        output = output.reshape(valid_x, valid_y)
        print output
        save_image(np.int32(np.round(output*255)), output_name)

def classify_image_np(image, model, patch_dims):
    layers = model.layers
    softmax_layer = layers[-1]
    layers = layers[:-1]
    valid_x = image.shape[0] - patch_dims[0] + 1
    valid_y = image.shape[1] - patch_dims[1] + 1

    pixels = [(x,y) for x in range(valid_x) for y in range(valid_y)]
    patchsize = patch_dims[0]*patch_dims[1]
    params_l = []
    for layer in layers:
        flattener = layer.layers[0]
        params = flattener.get_params()
        params = map(lambda x: x.get_value(), params)
        params_l.append(params)
    softmax_params = [softmax_layer.W.get_value(), softmax_layer.b.get_value()]
    
    inputs = np.zeros(patchsize, dtype=np.float32)
    classes = []
    for pixel in pixels:
        inputs = image[pixel[0]:pixel[0]+patch_dims[0], pixel[1]:pixel[1]+patch_dims[1]].ravel()
        temp = inputs
        for params in params_l:
            weights = params[0].transpose(); biases = params[1];
            output = np.dot(weights, temp) + biases
            output = rectify(output)
            temp = np.concatenate((output, inputs), axis=1)
        #softmax layer is getting input twice
        temp = np.concatenate((temp, inputs), axis=1)
        classes.append(softmax(temp, softmax_params))
    print classes
    return np.array(classes)

def rectify(in_array):
    vrectify = np.vectorize(lambda x: max(0, x)) 
    return vrectify(in_array)
    
def softmax(in_array, params):
    weights = params[0].transpose()
    biases = params[1]
    temp = np.dot(weights, in_array) + biases
    num = np.exp(temp[0]);
    den = num + np.exp(temp[1]);
    
    return num/den


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print "Usage: python classify.py <image_folder> <output_folder> <model_file>"
        sys.exit()
    image_path = sys.argv[1]
    output_path = sys.argv[2]
    model_file_name = sys.argv[3]
    images = sorted(glob.glob(image_path + "/*"))[0:2]
    output_names = [output_path.rstrip("/") + "/" + image_name.split("/")[-1].rstrip(".tif") + "_classified.tif" for image_name in images]
    classify(images, model_file_name, output_names)

