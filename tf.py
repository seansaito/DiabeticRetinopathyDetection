import tensorflow as tf
import numpy as np
from PIL import Image

import os
import csv
import datetime, time

from sklearn.cross_validation import train_test_split

def log(text):
    text = '[%s] %s' % (str(datetime.datetime.now()), text)
    print text

# load the images
# directory = "max-cropped"
# images = [Image.open(directory + "/" + f) for f in os.listdir(directory)]
#
# X = np.array([np.array(im) for im in images])
#
# # Groundtruths
# f_names = [f.split(".")[0] for f in os.listdir(directory)]
# fp = open("trainLabels.csv")
# reader = csv.DictReader(fp)
# label_store = {row["image"]: row["level"] for row in reader}
# fp.close()
#
# y = np.array([int(label_store[f]) for f in f_names])
#
# np.save("X.npy", X)
# np.save("y.npy", y)

X = np.load("X.npy")
y = np.load("y.npy")

X, X_test, y, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# Global variables
IMAGE_SIZE = 512
NUM_CHANNELS = 3
PIXEL_DEPTH = 255
NUM_LABELS = 5
BATCH_SIZE = 20
NUM_EPOCHS = 10
SEED = 66478
learning_rate = 1e-5

train_size = y.shape[0]

# Placeholder definitions
train_data_node = tf.placeholder(
      tf.float32,
      shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))

# Weights
conv1_weights = tf.Variable(tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                      stddev=0.1,
                      seed=SEED, dtype=tf.float32))
conv1_biases = tf.Variable(tf.zeros([32], dtype=tf.float32))
conv2_weights = tf.Variable(tf.truncated_normal(
  [5, 5, 32, 64], stddev=0.1,
  seed=SEED, dtype=tf.float32))
conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=tf.float32))
fc1_weights = tf.Variable(  # fully connected, depth 512.
  tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512],
                      stddev=0.1,
                      seed=SEED,
                      dtype=tf.float32))
fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=tf.float32))
fc2_weights = tf.Variable(tf.truncated_normal([512, NUM_LABELS],
                                            stddev=0.1,
                                            seed=SEED,
                                            dtype=tf.float32))
fc2_biases = tf.Variable(tf.constant(
  0.1, shape=[NUM_LABELS], dtype=tf.float32))

# Model definition
conv = tf.nn.conv2d(train_data_node,
                    conv1_weights,
                    strides=[1, 1, 1, 1],
                    padding='SAME')
# Bias and rectified linear non-linearity.
relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
# Max pooling. The kernel size spec {ksize} also follows the layout of
# the data. Here we have a pooling window of 2, and a stride of 2.
pool = tf.nn.max_pool(relu,
                      ksize=[1, 2, 2, 1],
                      strides=[1, 2, 2, 1],
                      padding='SAME')
conv = tf.nn.conv2d(pool,
                    conv2_weights,
                    strides=[1, 1, 1, 1],
                    padding='SAME')
relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
pool = tf.nn.max_pool(relu,
                      ksize=[1, 2, 2, 1],
                      strides=[1, 2, 2, 1],
                      padding='SAME')
# Reshape the feature map cuboid into a 2D matrix to feed it to the
# fully connected layers.
pool_shape = pool.get_shape().as_list()
reshape = tf.reshape(
    pool,
    [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
# Fully connected layer. Note that the '+' operation automatically
# broadcasts the biases.
hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
logits = tf.matmul(hidden, fc2_weights) + fc2_biases

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, train_labels_node))

regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
              tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
loss += 5e-4 * regularizers

optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss)
train_prediction = tf.nn.softmax(logits)

if __name__ == "__main__":
    with tf.Session() as sess:
        # Run all the initializers to prepare the trainable parameters.
        tf.initialize_all_variables().run()
        print('Initialized!')
        # Loop through training steps.
        for step in xrange(int(NUM_EPOCHS * train_size) // BATCH_SIZE):
          # Compute the offset of the current minibatch in the data.
          # Note that we could use better randomization across epochs.
          offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
          batch_data = X[offset:(offset + BATCH_SIZE), ...]
          batch_labels = y[offset:(offset + BATCH_SIZE)]
          # This dictionary maps the batch data (as a numpy array) to the
          # node in the graph it should be fed to.
          feed_dict = {train_data_node: batch_data,
                       train_labels_node: batch_labels}
          # Run the graph and fetch some of the nodes.
          _, l, predictions = sess.run(
              [optimizer, loss, train_prediction],
              feed_dict=feed_dict)


          log("Step %i, Loss %f" % (step, l))
