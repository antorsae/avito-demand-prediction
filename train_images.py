import argparse
import glob
import random
import pickle
import os
from os.path import join
from pathlib import Path
import numpy as np
import math
import re
#KERAS_BACKEND='tensorflow' python3 train_images.py  -p avg -cm ResNet152 -nfc -bf 4096 -fcm -uiw

from keras.optimizers import Adam, Adadelta, SGD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, Callback
from keras.models import load_model, Model
from keras.layers import concatenate, Lambda, Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, \
        BatchNormalization, Activation, GlobalAveragePooling2D, AveragePooling2D, Reshape, SeparableConv2D
from keras import backend as K
from keras.engine.topology import Layer
from keras.utils import to_categorical
import keras.losses
from multi_gpu_keras import multi_gpu_model
from clr_callback import CyclicLR

from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split

import imgaug as ia
from imgaug import augmenters as iaa
import jpeg4py as jpeg
from extra import *
from keras.applications import *
import inspect

from multiprocessing import Pool
from multiprocessing import cpu_count, Process, Queue, JoinableQueue, Lock
import sharedmem
import cv2

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

tf=K.tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

# yapf: disable
parser = argparse.ArgumentParser()
# general
parser.add_argument('--max-epoch', type=int, default=1000, help='Epoch to run')
parser.add_argument('-g', '--gpus', type=int, default=None, help='Number of GPUs to use')
parser.add_argument('-b', '--batch-size', type=int, default=48, help='Batch Size during training, e.g. -b 64')
parser.add_argument('-l', '--learning-rate', type=float, default=None, help='Initial learning rate')
parser.add_argument('-clr', '--cyclic_learning_rate', action='store_true', help='Use cyclic learning rate https://arxiv.org/abs/1506.01186')
parser.add_argument('-o', '--optimizer', type=str, default='adam', help='Optimizer to use in training -o adam|sgd|adadelta')
parser.add_argument('--amsgrad', action='store_true', help='Apply the AMSGrad variant of adam|adadelta from the paper "On the Convergence of Adam and Beyond".')

# architecture/model
parser.add_argument('-m', '--model', help='load hdf5 model including weights (and continue training)')
parser.add_argument('-w', '--weights', help='load hdf5 weights only (and continue training)')
parser.add_argument('-lo', '--loss', type=str, default='categorical_crossentropy', help='Loss function')
parser.add_argument('-do', '--dropout', type=float, default=0., help='Dropout rate for first FC layer')
parser.add_argument('-dol', '--dropout-last', type=float, default=0., help='Dropout rate for last FC layer')
parser.add_argument('-fc', '--fully-connected-layers', nargs='+', type=int, default=[512, 256], help='Specify FC layers, e.g. -fc 1024 512 256')
parser.add_argument('-bn', '--batch-normalization', action='store_true', help='Use batch normalization in FC layers')
parser.add_argument('-fca', '--fully-connected-activation', type=str, default='relu', help='Activation function to use in FC layers, e.g. -fca relu|selu|prelu|leakyrelu|elu|...')
parser.add_argument('-bf', '--bottleneck-features', type=int, default=16384, help='If classifier supports it, override number of bottleneck feautures (typically 2048)')
parser.add_argument('-cm', '--classifier', type=str, default='ResNet50', help='Base classifier model to use')
parser.add_argument('-pcs', '--print-classifier-summary', action='store_true', help='Print classifier model summary')
parser.add_argument('-uiw', '--use-imagenet-weights', action='store_true', help='Use imagenet weights (transfer learning)')
parser.add_argument('-p', '--pooling', type=str, default='avg', help='Type of pooling to use: avg|max|none')
parser.add_argument('-nfc', '--no-fcs', action='store_true', help='Dont add any FC at the end, just a softmax')
parser.add_argument('-pp', '--post-pooling', type=str, default=None, help='Add pooling layers after classifier, e.g. -pp avg|max')
parser.add_argument('-pps', '--post-pool-size', type=int, default=2, help='Pooling factor for pooling layers after classifier, e.g. -pps 3')

parser.add_argument('-fcm', '--freeze-classifier', action='store_true', help='Freeze classifier weights (useful to fine-tune FC layers)')
parser.add_argument('-fac', '--freeze-all-classifiers', action='store_true', help='Freeze all classifier (feature extractor) weights when using -id')


# test
parser.add_argument('-t', '--test', action='store_true', help='Test model and generate CSV/npy submission file')
parser.add_argument('-tt', '--test-train', action='store_true', help='Test model on the training set')
# yapf: enable

args = parser.parse_args()

training = not (args.test or args.test_train)

from tensorflow.python.client import device_lib


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


if args.gpus is None:
    args.gpus = len(get_available_gpus())

args.batch_size *= max(args.gpus, 1)

CROP_SIZE = 256
# data
N_CLASSES = 3067
MODEL_FOLDER = 'models'
os.makedirs(MODEL_FOLDER, exist_ok=True)

TRAIN_DATA = []
TRAIN_LABELS = []
LABELS = dict()

for line in open('image_top_1.csv').read().splitlines():
    TRAIN_DATA.append(line.split(',')[0])
    TRAIN_LABELS.append(int(line.split(',')[1]))
    LABELS[line.split(',')[0]] = line.split(',')[1]

print(len(TRAIN_DATA), len(TRAIN_LABELS))

IDX_TRAIN_SPLIT, IDX_VALID_SPLIT = train_test_split(
    TRAIN_DATA, test_size=0.1, random_state=SEED)

print('Training on {} samples'.format(len(IDX_TRAIN_SPLIT)))
print('Validating on {} samples'.format(len(IDX_VALID_SPLIT)))


# multiprocess worker to read items and put them in shared memory for consumer

def process_item_worker(worker_id, lock, shared_mem_X, shared_mem_y, jobs,
                        results, training):
    # make sure augmentations are different for each worker
    new_seed = worker_id
    np.random.seed(new_seed)
    random.seed(new_seed)

    while True:
        item = jobs.get()
        img, label, item = process_item(item, aug=training)
        is_good_item = False
        if img is not None:
            lock.acquire()
            shared_mem_X[worker_id, ...] = img
            shared_mem_y[worker_id, ...] = label
            is_good_item = True
        results.put((worker_id, is_good_item, item))


def gen(args, items, training=True):
    predict = False
    n_workers = (
        cpu_count() -
        1) if not predict else 1  # for prediction we need to guarantee order
    shared_mem_X = sharedmem.empty(
        (n_workers, CROP_SIZE, CROP_SIZE, 3), dtype=np.float32)
    shared_mem_y = sharedmem.empty(
        (n_workers, N_CLASSES), dtype=np.int32)

    locks = [Lock()] * n_workers
    jobs = Queue(args.batch_size * 4 if not predict else 1)
    results = JoinableQueue(args.batch_size * 2 if not predict else 1)

    [
        Process(
            target=process_item_worker,
            args=(worker_id, lock, shared_mem_X, shared_mem_y, jobs,
                  results, training)).start() for worker_id, lock in enumerate(locks)
    ]

    i = 0
    X = np.empty(
        (args.batch_size, CROP_SIZE, CROP_SIZE, 3),
        dtype=np.float32)
    y = np.empty(
        (args.batch_size, N_CLASSES),
        dtype=np.int32)

    while True:

        random.shuffle(items)

        batch_idx = 0

        items_done = 0

        while items_done < len(items):
            # fill the queue to make sure CPU is always busy
            while not jobs.full():
                item = items[i % len(items)]
                i += 1

                jobs.put(item)
                items_done += 1

            # loop over results and yield until no more resuls left
            get_more_results = True
            while get_more_results:
                worker_id, is_good_item, _item = results.get(
                )  # blocks/waits if None
                results.task_done()

                if is_good_item:
                    X[batch_idx], y[batch_idx] = shared_mem_X[
                        worker_id], shared_mem_y[worker_id]
                    locks[worker_id].release()
                    batch_idx += 1

                if batch_idx == args.batch_size:
                    yield (X, y)
                    batch_idx = 0

                get_more_results = not results.empty()

def preprocess_image(img):
    
    # find `preprocess_input` function specific to the classifier
    classifier_to_module = { 
        'NASNetLarge'       : 'nasnet',
        'NASNetMobile'      : 'nasnet',
        'DenseNet121'       : 'densenet',
        'DenseNet161'       : 'densenet',
        'DenseNet201'       : 'densenet',
        'InceptionResNetV2' : 'inception_resnet_v2',
        'InceptionV3'       : 'inception_v3',
        'MobileNet'         : 'mobilenet',
        'ResNet50'          : 'resnet50',
        'VGG16'             : 'vgg16',
        'VGG19'             : 'vgg19',
        'Xception'          : 'xception',

        'VGG16Places365'        : 'vgg16_places365',
        'VGG16PlacesHybrid1365' : 'vgg16_places_hybrid1365',

        'SEDenseNetImageNet121' : 'se_densenet',
        'SEDenseNetImageNet161' : 'se_densenet',
        'SEDenseNetImageNet169' : 'se_densenet',
        'SEDenseNetImageNet264' : 'se_densenet',
        'SEInceptionResNetV2'   : 'se_inception_resnet_v2',
        'SEMobileNet'           : 'se_mobilenets',
        'SEResNet50'            : 'se_resnet',
        'SEResNet101'           : 'se_resnet',
        'SEResNet154'           : 'se_resnet',
        'SEInceptionV3'         : 'se_inception_v3',
        'SEResNext'             : 'se_resnet',
        'SEResNextImageNet'     : 'se_resnet',

        'ResNet152'             : 'resnet152',
        'AResNet50'             : 'aresnet50',
        'AXception'             : 'axception',
        'AInceptionV3'          : 'ainceptionv3',
    }

    if args.classifier in classifier_to_module:
        classifier_module_name = classifier_to_module[args.classifier]
    else:
        classifier_module_name = 'xception'

    preprocess_input_function = getattr(globals()[classifier_module_name], 'preprocess_input')
    return preprocess_input_function(img.astype(np.float32))



def augment_soft(img):
    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    # Define our sequence of augmentation steps that will be applied to every image
    # All augmenters with per_channel=0.5 will sample one value _per image_
    # in 50% of all cases. In all other cases they will sample new values
    # _per channel_.
    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            # crop images by -5% to 10% of their height/width
            iaa.Crop(
                percent=(0, 0.2),
            ),
            iaa.Scale({"height": CROP_SIZE, "width": CROP_SIZE }),
        ],
        random_order=False
    )

    if img.ndim == 3:
        img = seq.augment_images(np.expand_dims(img, axis=0)).squeeze(axis=0)
    else:
        img = seq.augment_images(img)

    return img

def process_item(item, aug=False):
    ok = True
    try:
        img = jpeg.JPEG(item+'.jpg').decode()
    except:
        ok = False
    if not ok:
        return None, None, None
    if aug:
        img = augment_soft(img)
    else:
        img = cv2.resize(img, (CROP_SIZE, CROP_SIZE))
    label = LABELS[item]
    label = to_categorical(label, N_CLASSES) 
    img = preprocess_image(img)
    if len(img.shape) != 3:
        return None, None, None
    return img, label, item


if args.model:
    print("Loading model " + args.model)
    model = load_model(
        args.model,
        compile=False
        if not training or (args.learning_rate is not None) else True)
    model_basename = os.path.splitext(os.path.basename(args.model))[0]
    model_parts = model_basename.split('-')
    model_name = '-'.join(
        [part for part in model_parts if part not in ['epoch', 'val_acc']])
    last_epoch = int(
        list(filter(lambda x: x.startswith('epoch'), model_parts))[0][5:])
    print("Last epoch: {}".format(last_epoch))
    if args.learning_rate == None and training:
        dummy_model = model
        args.learning_rate = K.eval(model.optimizer.lr)
        print("Resuming with learning rate: {:.2e}".format(args.learning_rate))

elif True:

    classifier = globals()[args.classifier]
    print(args.classifier)
    kwargs = { \
        'include_top' : False,
        'weights'     : 'imagenet' if args.use_imagenet_weights else None,
        'input_shape' : (CROP_SIZE, CROP_SIZE, 3), 
        'pooling'     : args.pooling if args.pooling != 'none' else None,
     }

    classifier_args, _, _, _ = inspect.getargspec(classifier)

    if 'bottleneck_features' in classifier_args:
        kwargs['bottleneck_features'] = args.bottleneck_features

    classifier_model = classifier(**kwargs)

    if args.print_classifier_summary:
        classifier_model.summary()


    if args.learning_rate is None:
        args.learning_rate = 1e-4  # default LR unless told otherwise

    last_epoch = 0

    input_image = Input(shape=(CROP_SIZE, CROP_SIZE, 3),  name = 'image' )
    x = input_image

    x = classifier_model(x)

    if not args.no_fcs:
        dropouts = np.linspace(args.dropout, args.dropout_last,
                               len(args.fully_connected_layers))

        for i, (fc_layer, dropout) in enumerate(
                zip(args.fully_connected_layers, dropouts)):
            if args.batch_normalization:
                x = Dense(fc_layer, name='fc{}'.format(i))(x)
                x = BatchNormalization(name='bn{}'.format(i))(x)
                x = Activation(
                    args.fully_connected_activation,
                    name='act{}{}'.format(args.fully_connected_activation, i))(x)
            else:
                x = Dense(
                    fc_layer,
                    activation=args.fully_connected_activation,
                    name='fc{}'.format(i))(x)
            if dropout != 0:
                x = Dropout(
                    dropout, name='dropout_fc{}_{:04.2f}'.format(i, dropout))(x)

    if args.post_pooling is not None:
        if args.post_pooling == 'avg':
            x = AveragePooling2D(pool_size=args.post_pool_size)(x)
        elif args.post_pooling == 'max':
            x = MaxPooling2D(pool_size=args.post_pool_size)(x)

    x = Dense(N_CLASSES, name="logits")(x)

    prediction = Activation(activation="softmax", name="predictions")(x)

    model = Model(inputs=input_image, outputs=prediction)

    if args.freeze_classifier:
        for layer in model.layers:
            if isinstance(layer, Model):
                print("Freezing weights for classifier {}".format(layer.name))
                for classifier_layer in layer.layers:
                    classifier_layer.trainable = False
                if not args.freeze_all_classifiers:
                    break # otherwise freeze only first

    print(model.summary())

    if args.weights:
        print("Loading weights from {}".format(args.weights))
        model.load_weights(args.weights, by_name=True, skip_mismatch=True)
        match = re.search(r'([,A-Za-z_\d\.]+)-epoch(\d+)-.*\.hdf5',
                          args.weights)
        last_epoch = int(match.group(2))

if training:
    if args.optimizer == 'adam':
        opt = Adam(lr=args.learning_rate, amsgrad=args.amsgrad)
    elif args.optimizer == 'sgd':
        opt = SGD(
            lr=args.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    elif args.optimizer == 'adadelta':
        opt = Adadelta(lr=args.learning_rate, amsgrad=args.amsgrad)
    else:
        assert False

    loss = {'predictions': args.loss}

    model = multi_gpu_model(model, gpus=args.gpus)

    model.compile(
        optimizer=opt,
        loss=loss,
        metrics={
            'predictions': ['categorical_accuracy']
        })

    model_name = 'avito'
    mode = 'max'
    metric = "-val_acc{val_categorical_accuracy:.6f}"
    monitor = "val_categorical_accuracy"

    save_checkpoint = ModelCheckpoint(
        join(MODEL_FOLDER,
             model_name + "-epoch{epoch:03d}" + metric + ".hdf5"),
        monitor=monitor,
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode=mode,
        period=1)

    reduce_lr = ReduceLROnPlateau(
        monitor=monitor,
        factor=0.2,
        patience=5,
        min_lr=1e-9,
        epsilon=0.00001,
        verbose=1,
        mode=mode)

    clr = CyclicLR(
        base_lr=args.learning_rate / 4,
        max_lr=args.learning_rate,
        step_size=int(math.ceil(len(IDX_TRAIN_SPLIT) / args.batch_size)) * 1,
        mode='exp_range',
        gamma=0.99994)

    callbacks = [save_checkpoint]

    if args.cyclic_learning_rate:
        callbacks.append(clr)
    else:
        callbacks.append(reduce_lr)

    model.fit_generator(
        use_multiprocessing=False,
        generator=gen(args, IDX_TRAIN_SPLIT, True),
        steps_per_epoch=np.ceil(
            float(len(IDX_TRAIN_SPLIT)) / float(args.batch_size)) - 1,
        epochs=args.max_epoch,
        verbose=1,
        callbacks=callbacks,
        validation_data=gen(args, IDX_VALID_SPLIT, False),
        validation_steps=np.ceil(
            float(len(IDX_VALID_SPLIT)) / float(args.batch_size)) - 1,
        initial_epoch=last_epoch)

elif args.test or args.test_train:
    pass
