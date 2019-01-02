from model import *

class NeuralStyleTransfer:

    def __init__(self, weights_filename, weights_path = "./", download_weights = False):
        if download_weights:
            download_weights(WEIGHTS_URL, weights_path, weights_path)
        path = os.path.join(weights_path, weights_filename)
        self.model = Model(path).generate_model()
        print(self.model)

    def run(self):
        pass

    def content_cost(self, content_img, other_img):
        print(content_img.shape)

test = NeuralStyleTransfer(WEIGHTS_PATH)
