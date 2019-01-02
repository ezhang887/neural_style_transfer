import tensorflow as tf
import numpy as np
import scipy.io
import urllib.request
import os

WEIGHTS_URL = "http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat"
WEIGHTS_PATH = "./imagenet-vgg-verydeep-19.mat"

class Model:

    VGG19_LAYERS = [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4'
    ]

    def __init__(self, path):
        print("Loading weights...")
        self.weights = scipy.io.loadmat(path)["layers"][0]
        print("Done loading weights.")

    def get_weights(self, layer):
        idx = self.VGG19_LAYERS.index(layer)
        weights, bias = self.weights[idx][0][0][2][0]
        weights = np.transpose(weights, (1, 0, 2, 3))
        bias = bias.reshape(-1)
        return weights, bias

    def conv_2d(self, input, layer):
        weights, bias = self.get_weights(layer)
        return tf.nn.bias_add(tf.nn.conv2d(input, tf.constant(weights), strides = (1,1,1,1), padding = "SAME"), bias)

    def relu(self, input):
        return tf.nn.relu(input)
    
    def max_pool(self, input):
        return tf.nn.max_pool(input, ksize = [1,2,2,1], strides = [1,2,2,1], padding = "SAME")

    def generate_model(self):
        model = {}
        model['input'] = tf.Variable(np.zeros((1, 224, 224, 3)), dtype = 'float32')
        prev_layer = "input"
        for layer in self.VGG19_LAYERS:
            layer_type = layer[:4]
            if layer_type == "conv":
                model[layer] = self.conv_2d(model[prev_layer], layer)
            elif layer_type == "pool":
                model[layer] = self.max_pool(model[prev_layer])
            elif layer_type == "relu":
                model[layer] = self.relu(model[prev_layer])
            prev_layer = layer
        return model

def download_weights(url, filename, path = "./"):
    if not os.path.isfile(filename): 
        try:
            os.makedirs(path)
        except Exception:
            pass
        path = os.path.join(path, filename)
        print("Saving weights from {} to {}".format(url, path))
        urllib.request.urlretrieve(url, path)
