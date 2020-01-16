#!/usr/bin/env python3

from nst_utils import *
from model import Model
import tensorflow as tf
import os
import subprocess

def run():
    size = (360, 480)
    content_img = preprocess(read_image("nyc.jpg"), size)
    style_img = preprocess(read_image("starry_night.jpg"), size)
    initial_output = generate_initial_output(content_img)

    model = Model("imagenet-vgg-verydeep-19.mat").generate_model(size)

    alpha = 10
    beta = 50

    iterations = 300

    output_dir = "output/"

    style_layers = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]

    with tf.Session() as sess:
        sess.run(model["input"].assign(content_img))
        content_cost = compute_content_cost(sess.run(model["conv4_2"]), model["conv4_2"])

        sess.run(model["input"].assign(style_img))
        style_cost = compute_style_cost(sess, model, style_layers)

        total_cost = compute_total_cost(content_cost, style_cost, alpha, beta)

        optimizer = tf.train.AdamOptimizer(2.0)
        step = optimizer.minimize(total_cost)

        sess.run(tf.global_variables_initializer())
        sess.run(model["input"].assign(initial_output))

        for i in range(1, iterations+1):
            sess.run(step)
            if i%10 == 0:
                output = sess.run(model["input"])
                print("Iteration: " + str(i))
                print("Style cost: " + str(sess.run(style_cost)))
                print("Content cost: " + str(sess.run(content_cost)))

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                output_name = output_dir + str(i) + ".jpg"
                #cv2.imshow(output_name, unpreprocess(output))
                cv2.imwrite(output_name, unpreprocess(output))

if __name__ == "__main__":
    run()
