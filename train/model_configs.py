import random
import numpy as np 
import tensorflow as tf

class resnet_config():
    def __init__(self):

        # model and training config
        self.flattened_input = False
        self.near_pos_multiple = 0.2
        self.rand_neg_multiple = 19.8 
        self.num_train_sets = 10
        self.hyper_search = False
        self.model_name = 'resnet'

        # hyper params
        self.chaining_dists = [i for i in range(1,11)]
        self.thresholds = np.linspace(0.5, 0.9, 10)
        self.tolerance_seconds = 5
        self.n_iterations = 20
        self.batch_size = 128
        self.optimizer = 'adam'
        self.learning_rate = 1e-3
        self.pos_weight = 1
        self.loss = get_weighted_bce(self.pos_weight)
        self.metrics=['accuracy']

class ff_config():
    def __init__(self):

        # model and training config
        self.flattened_input = True
        self.near_pos_multiple = 0.2
        self.rand_neg_multiple = 19.8 
        self.num_train_sets = 10
        self.hyper_search = False
        self.model_name = 'feed_forward'

        # hyper params
        self.chaining_dists = [i for i in range(1,11)]
        self.thresholds = np.linspace(0.5, 0.9, 10)
        self.tolerance_seconds = 5
        self.n_iterations = 20
        self.batch_size = 128
        self.optimizer = 'adam'
        self.learning_rate = 1e-3
        self.hidden_layers = [32,20]
        self.l2_reg = 0.0
        self.batch_norm = True
        self.activation = 'relu'
        self.output_activation = 'sigmoid'
        self.pos_weight = 1
        self.loss = get_weighted_bce(self.pos_weight)
        self.metrics=['accuracy']

class ff_search_config():
    def __init__(self):

        # model and training config
        self.flattened_input = True
        self.near_pos_multiple = 0.2
        self.rand_neg_multiple = 19.8 
        self.num_train_sets = 10 
        self.hyper_search = True
        self.model_name = 'feed_forward'

        # hyper params
        self.chaining_dists = [i for i in range(1,11)]
        self.thresholds = np.linspace(0.5, 0.9, 10)
        self.tolerance_seconds = 5
        self.n_iterations = 20
        self.batch_size = 128
        self.optimizer = 'adam'
        exp = -(2*np.random.random()+2)
        self.learning_rate = 10**exp
        h_layers = random.sample([[32, 20], [64, 32, 20], [128, 64, 32, 20]], 1)[0]
        self.hidden_layers = h_layers
        reg = random.sample([0., 1e-4, 1e-2, 1.], 1)[0]
        self.l2_reg = reg
        self.batch_norm = True
        self.activation = 'relu'
        self.output_activation = 'sigmoid'
        # self.loss = 'binary_crossentropy'
        weight = random.sample([8, 18, 28], 1)[0]
        self.pos_weight = weight
        self.loss = get_weighted_bce(self.pos_weight)
        self.metrics=['accuracy']


def get_config(name):
    if name =='feed_forward':
        return ff_config()
    elif name =='resnet':
        return resnet_config()
    elif name == 'feed_forward_search':
        return ff_search_config()

def get_weighted_bce(pos_weight):
    def weighted_bce(y_true,y_pred):
        return tf.nn.weighted_cross_entropy_with_logits(
                y_true,
                y_pred,
                pos_weight,
            )
    
    return weighted_bce
