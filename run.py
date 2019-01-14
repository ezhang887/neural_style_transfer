#!/usr/bin/env python3
from nst_utils import *
from model import Model
import tensorflow as tf

def run():
    size = (360, 480)
    content_img = preprocess(read_image("nyc.jpg"), size)
    style_img = preprocess(read_image("starry_night.jpg"), size)

    model = Model("imagenet-vgg-verydeep-19.mat").generate_model(size)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(model["input"].assign(content_img))

if __name__ == "__main__":
    run()
