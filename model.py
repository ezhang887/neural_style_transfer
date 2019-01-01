import tensorflow as tf
import scipy.io
import urllib.request
import os

WEIGHTS_URL = "http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat"
WEIGHTS_PATH = "./imagenet-vgg-verydeep-19.mat"

class Model:

    def __init__(self, path):
        print("Loading weights...")
        self.vgg = scipy.io.loadmat(path)["layers"]
        print("Done loading weights.")

    def get_weights(self, layer):
        params = self.vgg[0][layer][0][0][2]
        weights = params[0][0]
        bias = params[0][1]

    def conv_2d(self, input, layer):
        weights, bias = get_weights(self, layer)
        return tf.nn.bias_add(tf.nn.conv2d(input, tf.constant(weights), strides = (1,1,1,1), padding = "SAME"), bias)

    def relu(self, input):
        return tf.nn.relu(input)
    
    def max_pool(self, input):
        return tf.nn.max_pool(input, ksize = [1,2,2,1], strides = [1,2,2,1], padding = "SAME")

    def generate_model(self):
        model = {}

        return model

def download_weights(url, filename):
    if not os.path.isfile(filename): 
        print("Saving weights from {} to {}".format(url, filename))
        urllib.request.urlretrieve(url, filename)

download_weights(WEIGHTS_URL, WEIGHTS_PATH)
Model(WEIGHTS_PATH)
