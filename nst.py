from model import *

class NeuralStyleTransfer:

    """
    Arguments:
    weights_filename -- a string representing the file name of the downloaded weights
    weights_path -- a string representing the relative path to the weights file, default is the current directory
    download_weights -- a boolean set to True to download the weights from the internet, otherwise (and default) is False
    """
    def __init__(self, weights_filename, weights_path = "./", download_weights = False):
        if download_weights:
            download_weights(WEIGHTS_URL, weights_path, weights_path)
        path = os.path.join(weights_path, weights_filename)
        self.model = Model(path).generate_model()

    """
    Computes the content cost as specified in the research paper
    
    Arguments:
    content_img -- a Tensor of dimension (1, height, width, channels), representing the hidden layer activations of the input content image
    generated_img -- a Tensor of dimension (1, height, width, channels), representing the hidden layer activations of the ouput image
    """
    def content_cost(self, content_img, generated_img):
        _, h, w, c = content_img.get_shape().as_list()
        cost = 1/(4*h*w*c)*tf.reduce_sum(tf.square(content_img-generated_img))
        return cost

    """
    Computes the style cost as specified in the research paper

    Arguments:
    style_img -- a Tensor of dimension (1, height, width, channels), representing the hidden layer activations of the input style image
    generated_img -- a Tensor of dimension (1, height, width, channels), representing the hidden layer activations of the output image
    """
    def style_cost(self, style_img, generated_img):
        _, h, w, c = style_img.get_shape().as_list()

        style_img = tf.transpose(tf.reshape(style_img, (h*w, c)))
        generated_img = tf.transpose(tf.reshape(generated_img , (h*w, c)))

        gram_style = self.gram_matrix(style_img)
        gram_generated = self.gram_matrix(generated_img)

        cost = 1/(4*c**2*(h*w)**2)*tf.reduce_sum(tf.square(gram_style - gram_generated))
        return cost

    """
    Computes the gram matrix of a specified matrix

    Arguments:
    mat -- a Tensor of dimension (channels, height*width)

    Returns a Tensor of dimension(channels, channels)
    """
    def gram_matrix(self, mat):
        return tf.matmul(mat, tf.transpose(mat))

    """
    Computes the total cost function of neural style trnasfer

    Arguments:
    content_cost -- a float representing the content_cost
    style_cost -- a float representing the style_cost
    alpha -- a float representing the hyperparameter for the content_cost
    beta -- a float representing the hyperparameter for the style cost
    """
    def total_cost(self, content_cost, style_cost, alpha, beta):
        return alpha*content_cost + beta*style_cost

    def run(self, content_img, style_img):
        pass

nst = NeuralStyleTransfer(WEIGHTS_PATH)
