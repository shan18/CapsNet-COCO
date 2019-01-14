import os
import h5py
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K
from keras import layers, models, optimizers, callbacks

from capsule_layers import CapsuleLayer, PrimaryCap, Length, Mask

K.set_image_data_format('channels_last')


def CapsNet(input_shape, n_class, routings):
    """
    A Capsule Network on MSCOCO.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape)

    # Layers 1-3: Just some conventional Conv2D layers
    conv1 = Conv2D(filters=96, kernel_size=13, strides=4, padding='valid', activation='relu', name='conv1')(x)
    conv2 = Conv2D(filters=96, kernel_size=5, strides=2, padding='valid', activation='relu', name='conv2')(conv1)
    conv3 = Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv3')(conv2)

    # Layer 4: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primary_caps = PrimaryCap(conv3, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    category_caps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
                             name='category_caps')(primary_caps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(category_caps)

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([category_caps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(category_caps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=16*n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])

    # manipulate model
    noise = layers.Input(shape=(n_class, 16))
    noised_category_caps = layers.Add()([category_caps, noise])
    masked_noised_y = Mask()([noised_category_caps, y])
    manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))

    return train_model, eval_model, manipulate_model


def margin_loss(y_true, y_pred):
    """
    Margin loss.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


def train(model, data, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data

    # callbacks
    log = callbacks.CSVLogger(args['save_dir'] + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args['save_dir'] + '/tensorboard-logs',
                               batch_size=args['batch_size'], histogram_freq=int(args['debug']))
    checkpoint = callbacks.ModelCheckpoint(args['save_dir'] + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args['lr'] * (args['lr_decay'] ** epoch))
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1)

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args['lr']),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args['lam_recon']],
                  metrics={'capsnet': 'accuracy'})

    # Training
    model.fit(
        [x_train, y_train],
        [y_train, x_train],
        batch_size=args['batch_size'],
        epochs=args['epochs'],
        validation_data=[[x_test, y_test], [y_test, x_test]],
        callbacks=[log, tb, checkpoint, lr_decay, early_stop]
    )

    model.save_weights(args['save_dir'] + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args['save_dir'])

    from utils import plot_log
    plot_log(args['save_dir'] + '/log.csv', show=True)

    return model


def load_coco(dataset_file, map_file):
    """
    Load preprocessed MSCOCO 2017 dataset
    """
    print('\nLoading dataset...')
    h5f = h5py.File(dataset_file, 'r')
    x = h5f['x'][:]
    y = h5f['y'][:]
    h5f.close()

    split = int(x.shape[0] * 0.8)  # 80% of data is assigned to the training set
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    with open(map_file, 'rb') as mapping:
        category_id_map = pickle.load(mapping)
    id_category = category_id_map['id_category']
    print('Done.')

    return (x_train, y_train), (x_test, y_test), id_to_category


if __name__ == "__main__":

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on MSCOCO 2017.")
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--lr', default=0.001, type=float, help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.392, type=float, help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--debug', action='store_true', help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument(
        '--dataset_file', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'capsnet_train_data.h5'),
        help='File having the preprocessed dataset'
    )
    parser.add_argument(
        '--map_file', default=os.path.join(os.path.dirname(os.path.realpath(__file__)), '../dataset/coco_raw.pickle'),
        help='File having the id to category map'
    )
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data
    (x_train, y_train), (x_test, y_test), id_to_category = load_coco(args.dataset_file, args.map_file)

    # define model
    model, eval_model, manipulate_model = CapsNet(
        input_shape=x_train.shape[1:],
        n_class=y_train.shape[1],
        routings=args.routings
    )
    model.summary()

    # train
    train(model=model, data=((x_train, y_train), (x_test, y_test)), args=vars(args))
