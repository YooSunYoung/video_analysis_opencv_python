import tensorflow as tf
import os
import numpy as np
import cv2
import random
import logging
tf.get_logger().setLevel(logging.ERROR)


def preprocess(directory_path="data/eye_images/mrlEyes_2018_01"):
    if os.path.exists(directory_path) is False:
        print("Please download and extract the eye images and check if they are in the right path.")
        return 0
    dir_list = os.listdir(directory_path)
    image_dictionary = {}
    for dir in dir_list:
        dir = os.path.join(directory_path, dir)
        image_list = os.listdir(dir)
        for image in image_list:
            info = image.split("_")
            if int(info[3]) == 1: continue  # if wearing glasses
            if int(info[5]) == 2: continue  # if high reflection
            if int(info[6]) == 0: continue  # if lighting condition in bad
            image_dictionary[os.path.join(dir, image)] = int(info[4])

    resized_image_dictionary = {}

    image_label_list = []

    for image in image_dictionary.keys():
        open_closed = image_dictionary[image]
        img = cv2.imread(image, 0)
        if img is None: continue
        if img.shape[0] < 50 or img.shape[1] < 50: continue  # discard too small images
        img = cv2.resize(img, (50, 50), interpolation=cv2.INTER_AREA)
        resized_image_dictionary[image] = open_closed
        image_label_list.append([img, open_closed])
        image_label_list.append([np.where(img+img > 255, 255, img+img), open_closed])
        image_label_list.append([img+img, open_closed])
        image_label_list.append([img+img*0.5, open_closed])
        image_label_list.append([img-img*0.1, open_closed])
        image_label_list.append([img-img*0.2, open_closed])
        image_label_list.append([img-img*0.3, open_closed])

    open_eyes = list(filter(lambda x: x[1] == 1, image_label_list))
    closed_eyes = list(filter(lambda x: x[1] == 0, image_label_list))
    random.shuffle(open_eyes)
    random.shuffle(closed_eyes)
    train_image_list = open_eyes[:45000]
    train_image_list.extend(closed_eyes[:45000])
    test_image_list = open_eyes[45000:48000]
    test_image_list.extend(closed_eyes[45000:48000])
    random.shuffle(train_image_list)
    random.shuffle(test_image_list)

    train_images = np.array([x[0] for x in train_image_list])
    train_images = train_images / 255
    train_images = train_images.reshape((-1, 50, 50, 1))
    train_labels = np.array([[float(x[1])] for x in train_image_list])

    with open('data/eye_images/train_image.npy', 'wb') as f:
        np.save(f, train_images)
    with open('data/eye_images/train_label.npy', 'wb') as f:
        np.save(f, train_labels)

    test_images = np.array([x[0] for x in test_image_list])
    test_images = test_images / 255
    test_images = test_images.reshape((-1, 50, 50, 1))
    test_labels = np.array([[float(x[1])] for x in test_image_list])

    with open('data/eye_images/test_image.npy', 'wb') as f:
        np.save(f, test_images)
    with open('data/eye_images/test_label.npy', 'wb') as f:
        np.save(f, test_labels)


def simple_model():
    cnn_1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(50, 50, 1))
    mxp_1 = tf.keras.layers.MaxPooling2D(2, 2)
    cnn_2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')
    mxp_2 = tf.keras.layers.MaxPooling2D(2, 2)
    cnn_3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
    mxp_3 = tf.keras.layers.MaxPooling2D(2, 2)
    cnn_4 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')
    mxp_4 = tf.keras.layers.MaxPooling2D(2, 2)
    flat = tf.keras.layers.Flatten()
    dense1 = tf.keras.layers.Dense(512, activation='relu')
    dense2 = tf.keras.layers.Dense(128, activation='relu')
    output = tf.keras.layers.Dense(1, activation='sigmoid')
    model = tf.keras.models.Sequential([
        cnn_1, mxp_1,
        cnn_2, mxp_2,
        cnn_3, mxp_3,
        #cnn_4, mxp_4,
        flat,
        dense1,
        dense2,
        output
    ])
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5)])
    model.summary()
    return model


def simple_model_tensorflow1(X, training=False):
    nodes_fc1 = 512
    # CONVOLUTION LAYER 1
    # Weights for layer 1
    w_1 = tf.Variable(tf.truncated_normal([11, 11, 1, 48], stddev=0.01))
    # Bias for layer 1
    b_1 = tf.Variable(tf.constant(0.0, shape=[[11, 11, 1, 48][3]]))
    # Applying convolution
    c_1 = tf.nn.conv2d(X, w_1, strides=[1, 1, 1, 1], padding='VALID')
    # Adding bias
    c_1 = c_1 + b_1
    # Applying RELU
    c_1 = tf.nn.relu(c_1)

    # POOLING LAYER1
    p_1 = tf.nn.max_pool(c_1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    # CONVOLUTION LAYER 2
    # Weights for layer 2
    w_2 = tf.Variable(tf.truncated_normal([5, 5, 48, 96], stddev=0.01))
    # Bias for layer 2
    b_2 = tf.Variable(tf.constant(1.0, shape=[[5, 5, 48, 96][3]]))
    # Applying convolution
    c_2 = tf.nn.conv2d(p_1, w_2, strides=[1, 1, 1, 1], padding='SAME')
    # Adding bias
    c_2 = c_2 + b_2
    # Applying RELU
    c_2 = tf.nn.relu(c_2)

    # POOLING LAYER2
    p_2 = tf.nn.max_pool(c_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flattening
    flattened = tf.reshape(p_2, [-1, 9 * 9 * 96])

    # Fully Connected Layer 1
    # Getting input nodes in FC layer 1
    input_size = int(flattened.get_shape()[1])
    # Weights for FC Layer 1
    w1_fc = tf.Variable(tf.truncated_normal([input_size, nodes_fc1], stddev=0.01))
    # Bias for FC Layer 1
    b1_fc = tf.Variable(tf.constant(1.0, shape=[nodes_fc1]))
    # Summing Matrix calculations and bias
    s_fc1 = tf.matmul(flattened, w1_fc) + b1_fc
    # Applying RELU
    s_fc1 = tf.nn.relu(s_fc1)

    # Dropout Layer is not supported for vitis-ai

    # Fully Connected Layer 3
    # Weights for FC Layer 3
    w3_fc = tf.Variable(tf.truncated_normal([nodes_fc1, 1], stddev=0.01))
    # Bias for FC Layer 3b3_fc = tf.Variable( tf.constant(1.0, shape=[output_classes] ) )
    b3_fc = tf.Variable(tf.constant(1.0, shape=[1]))
    y_pred = tf.matmul(s_fc1, w3_fc) + b3_fc

    if training: return y_pred
    y_pred = tf.nn.relu(y_pred, name="final_output")
    return y_pred


def train():
    train_images = np.load('data/eye_images/train_image.npy')
    train_labels = np.load('data/eye_images/train_label.npy')
    test_images = np.load('data/eye_images/test_image.npy')
    test_labels = np.load('data/eye_images/test_label.npy')

    model = simple_model()
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='data/model')
    model.fit(train_images, train_labels, batch_size=64, epochs=12,
              validation_data=(test_images, test_labels),
              callbacks=[cp_callback])
    model.evaluate(test_images, test_labels)
    model.save("data/model")
    del model


def test():
    model = simple_model()
    test_images = np.load('data/eye_images/test_image.npy')
    test_labels = np.load('data/eye_images/test_label.npy')
    model.load_weights('data/model')
    loss, acc = model.evaluate(test_images, test_labels, verbose=2)
    print("accuracy : {:5.2f}%".format(100*acc))

if __name__ == "__main__":
    # preprocess()
    # model = simple_model()
    train()
    # test()

