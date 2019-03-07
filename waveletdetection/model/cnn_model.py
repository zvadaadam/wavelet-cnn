import tensorflow as tf
from waveletdetection.model.base_model import ModelBase


class CNNModel(ModelBase):

    def __init__(self, config):
        super(CNNModel, self).__init__(config)

        self.loss = None
        self.acc = None
        self.opt = None
        self.logits = None
        self.x = None
        self.y = None

        self.init_placeholders()

    def init_placeholders(self):

        signal_length = self.config.signal_length()
        max_scale = self.config.max_scale()
        num_classes = self.config.num_classes()

        self.x = tf.placeholder(tf.float32, [None, max_scale - 1, signal_length, 1])
        self.y = tf.placeholder(tf.float32, [None, num_classes])

    def build_model(self, inputs):

        print('Building computation graph...')

        x = inputs['x']
        y = inputs['y']

        print(y.get_shape())

        batch_size = self.config.batch_size()
        num_layers = self.config.num_layers()
        num_classes = self.config.num_classes()

        cnn_output = self.build_cnn_layers(num_layers, x)

        # flatten cnn output for fully connected layer
        feature_dim = cnn_output.get_shape()[1:4].num_elements()
        cnn_output = tf.reshape(cnn_output, [-1, feature_dim])

        # fully connected layer 1 - reducer
        fc_output = self.fully_connected(cnn_output, 1024, scope_name='fully_conncted_1')

        # fully connected layer 2 - logits
        self.logits = self.fully_connected(fc_output, num_classes, scope_name='fully_conncted_2')

        print(self.logits.get_shape())

        self.loss = self.loss_function(self.logits, y)

        self.opt = self.optimizer(self.loss)

        self.acc = self.accuracy(self.logits, y)

    def build_cnn_layers(self, num_layers, x):

        input = x

        for layer in range(num_layers):

            conv = self.conv_relu(input, filters=32, filter_size=5, stride=1,
                                  padding='SAME', scope_name=f'conv{layer}')

            print(conv.get_shape())

            pool = self.maxpool(conv, ksize=2, stride=2, scope_name=f'pool{layer}')

            print(pool.get_shape())

            input = pool

        return input

    def conv_relu(self, inputs, filters, filter_size, stride, padding, scope_name='conv'):

        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
            input_channels = inputs.shape[-1]

            weights = tf.get_variable('weights', [filter_size, filter_size, input_channels, filters],
                            initializer=tf.truncated_normal_initializer())

            biases = tf.get_variable('biases', [filters], initializer=tf.random_normal_initializer())

            conv = tf.nn.conv2d(inputs, weights, strides=[1, stride, stride, 1], padding=padding)

        return tf.nn.relu(conv + biases, name=scope.name)

    def maxpool(self, inputs, ksize, stride, padding='VALID', scope_name='pool'):

        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
            pool = tf.nn.max_pool(inputs, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1],
                                  padding=padding)
        return pool

    def fully_connected(self, inputs, num_outputs, scope_name='fully_conncted'):

        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
            input_dim = inputs.shape[-1]
            print(input_dim)
            w = tf.get_variable('weights', [input_dim, num_outputs],
                                initializer=tf.truncated_normal_initializer())
            b = tf.get_variable('biases', [num_outputs],
                                initializer=tf.constant_initializer(0.0))
            logit = tf.matmul(inputs, w) + b

        return logit

    def loss_function(self, logits, y):

        with tf.name_scope('loss'):
            entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
            loss = tf.reduce_mean(entropy, name='mean_loss')

        return loss

    def accuracy(self, logits, y):

        with tf.name_scope('loss'):
            acc = tf.equal(tf.argmax(logits, axis=1), tf.argmax(y, axis=1))
            acc = tf.reduce_mean(tf.cast(acc, tf.float32))

        return acc

    def optimizer(self, loss):
        return tf.train.AdamOptimizer().minimize(loss)

