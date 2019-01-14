from model import *
import cv2

IMAGENET_MEANS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3)) 

""" Computes the content cost as specified in the research paper

Arguments:
content_img -- a Tensor of dimension (1, height, width, channels), representing the hidden layer activations of the input content image
generated_img -- a Tensor of dimension (1, height, width, channels), representing the hidden layer activations of the ouput image
"""
def content_cost(content_img, generated_img):
    _, h, w, c = content_img.get_shape().as_list()
    cost = 1/(4*h*w*c)*tf.reduce_sum(tf.square(content_img-generated_img))
    return cost

"""
Computes the style cost for a single style layer

Arguments:
style_img -- a Tensor of dimension (1, height, width, channels), representing the hidden layer activations of the input style image
generated_img -- a Tensor of dimension (1, height, width, channels), representing the hidden layer activations of the output image
"""
def style_layer_cost(style_img, generated_img):
    _, h, w, c = style_img.get_shape().as_list()

    style_img = tf.transpose(tf.reshape(style_img, (h*w, c)))
    generated_img = tf.transpose(tf.reshape(generated_img , (h*w, c)))

    gram_style = gram_matrix(style_img)
    gram_generated = gram_matrix(generated_img)

    cost = 1/(4*c**2*(h*w)**2)*tf.reduce_sum(tf.square(gram_style - gram_generated))
    return cost

"""
Computes the style cost through all the layers
"""
def style_cost(sess, model, style_layers):
    style_cost = 0
    num_layers = len(style_layers)
    weight = 1.0/num_layers
    for layer in style_layers:
        out = model[layer]
        style_activations = sess.run(out)
        generated_activation = out
        style_cost += weight*style_layer_cost(style_activations, generated_activation)
    return style_cost

"""
Computes the gram matrix of a specified matrix

Arguments:
mat -- a Tensor of dimension (channels, height*width)

Returns a Tensor of dimension(channels, channels)
"""
def gram_matrix(mat):
    return tf.matmul(mat, tf.transpose(mat))

"""
Computes the total cost function of neural style transfer

Arguments:
content_cost -- a float representing the content_cost
style_cost -- a float representing the style_cost
alpha -- a float representing the hyperparameter for the content_cost
beta -- a float representing the hyperparameter for the style cost
"""
def total_cost(content_cost, style_cost, alpha, beta):
    return alpha*content_cost + beta*style_cost

"""
Preprocesses an input image

Arguments:
image -- a numpy array representing an image
"""
def preprocess(image):
    #VGG19 input image size is (1,h,w,c)
    image = cv2.resize(image, (224,224))
    image = np.reshape(image, ((1,) + image.shape))
    #subtract out imagenet means for data to be centered around 0
    image = image - IMAGENET_MEANS
    return image

"""
Unprocesses an image for display

Arguments:
image -- a numpy array representing an image
"""
def unpreprocess(image):
    image = image + IMAGENET_MEANS
    image = image[0]
    image = np.clip(image[0], 0, 255)
    return image

"""
Generates a noise image based on the original content as the initial output

Arguments:
image -- a numpy array representing the content image
"""
def generate_initial_output(content_img):
    bound = 100
    noise = np.random.uniform(-bound, bound, content_img.shape).astype("float32")
    ratio = 0.85
    rv = noise*ratio + content_img*(1-ratio)
    return rv
