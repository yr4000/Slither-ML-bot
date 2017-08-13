import tensorflow as tf

OUTPUT_DIM = 8
INPUT_DIM = 400
SQRT_INPUT_DIM  =20 #IN ORDER TO RESHAPE INTO TENSOR
CONV_WINDOW_SIZE = 5
NUM_OF_CHANNELS_LAYER1 = 1
NUM_OF_CHANNELS_LAYER2 = 16
NUM_OF_CHANNELS_LAYER3 = 32
SIZE_OF_FULLY_CONNECTED_LAYER = 256



# TODO: what is dropout?!
keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

# build the computational graph
states = tf.placeholder(tf.int8, [None, INPUT_DIM], name='states')
actions = tf.placeholder(tf.int8, [None, OUTPUT_DIM], name='chosen_actions')
accu_rewards = tf.placeholder(tf.float32, [None, OUTPUT_DIM], name='accumulated_rewards')

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def convolutional_neural_network(x):
    weights = {'W_conv1': tf.Variable(tf.random_normal([CONV_WINDOW_SIZE , CONV_WINDOW_SIZE , NUM_OF_CHANNELS_LAYER1 , NUM_OF_CHANNELS_LAYER2])),
               'W_conv2': tf.Variable(tf.random_normal([CONV_WINDOW_SIZE , CONV_WINDOW_SIZE , NUM_OF_CHANNELS_LAYER2, NUM_OF_CHANNELS_LAYER3])),
               'W_fc': tf.Variable(tf.random_normal([CONV_WINDOW_SIZE  * CONV_WINDOW_SIZE  * NUM_OF_CHANNELS_LAYER3, SIZE_OF_FULLY_CONNECTED_LAYER])),
               'out': tf.Variable(tf.random_normal([SIZE_OF_FULLY_CONNECTED_LAYER, OUTPUT_DIM]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([NUM_OF_CHANNELS_LAYER2])),
              'b_conv2': tf.Variable(tf.random_normal([NUM_OF_CHANNELS_LAYER3])),
              'b_fc': tf.Variable(tf.random_normal([SIZE_OF_FULLY_CONNECTED_LAYER])),
              'out': tf.Variable(tf.random_normal([OUTPUT_DIM]))}

    x = tf.reshape(x, shape=[-1, SQRT_INPUT_DIM, SQRT_INPUT_DIM, NUM_OF_CHANNELS_LAYER1])

    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2, [-1, CONV_WINDOW_SIZE * CONV_WINDOW_SIZE * NUM_OF_CHANNELS_LAYER3])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    score = tf.matmul(fc, weights['out']) + biases['out']
    probability = tf.nn.softmax(score)
    return probability

'''
def apply_neural_network(state, score, snake):
    prediction = convolutional_neural_network(state)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    #hm_epochs = 10
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
'''