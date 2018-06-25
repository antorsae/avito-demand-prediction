import argparse
import random
import numpy as np
import pandas as pd
from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Input
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from geomean_layer import GeomeanLayer
from multi_gpu_keras import multi_gpu_model
from debug_callback import DebugCallback

from tensorflow.python.client import device_lib


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


gpus = len(get_available_gpus())

tf = K.tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)
K.set_session(sess)

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument('-mep', '--max-epoch', type=int, default=3, help='Epoch to run')
parser.add_argument('-b', '--batch-size', type=int, default=None, help='Batch Size during training, e.g. -b 2')
parser.add_argument('-g', '--gpus', type=int, default=1, help='GPUs')
parser.add_argument('-l', '--learning-rate', type=float, default=1e-3, help='Initial learning rate')

parser.add_argument('-m', '--model', help='load hdf5 model (and continue training)')
parser.add_argument('-w', '--weights', help='load hdf5 weights from model (and continue training)')

parser.add_argument('-d', '--debug', action='store_true', help='Debug')
parser.add_argument('-t', '--test', action='store_true', help='Test on test set')
# yapf: enable

a = parser.parse_args()

if a.batch_size is None:
    a.batch_size = 1024

if a.gpus is not None:
    gpus = a.gpus

a.batch_size *= gpus

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

df_x_train = pd.read_csv('train_oof_predictions_stack.csv')
df_y_train = pd.read_feather('df_y_train')
df_test = pd.read_csv('test_predictions_stack.csv')

X = np.clip(df_x_train.values, 0, 1)
Y = df_y_train['deal_probability'].values
X_test = np.clip(df_test.fillna(0.0).values, 0, 1)

def mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=None)


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=None))


def get_model():
    input = Input(shape=(X.shape[1], ), name='input')
    x = input
    weighted_geomean = GeomeanLayer()(x)
    model = Model(inputs=input, outputs=weighted_geomean)
    return model


def gen(idx, valid=False, X=None, Y=None):

    x = np.zeros((a.batch_size, X.shape[1]), dtype=np.float32)
    y = np.zeros((a.batch_size, 1), dtype=np.float32)

    batch = 0
    i = 0
    while True:
        if i == len(idx):
            i = 0
            if not valid and (Y is not None):
                random.shuffle(idx)

        x[batch, ...] = X[idx[i]]
        if Y is not None:
            y[batch, ...] = Y[idx[i]]

        batch += 1
        i += 1

        if batch == a.batch_size:

            assert not np.any(np.isnan(x))

            _x = np.copy(x)

            if Y is not None:
                assert not np.any(np.isnan(y))
                yield _x, np.copy(y)
            else:
                yield _x

            batch = 0


if a.model:
    model = load_model(a.model, compile=False)
else:
    model = get_model()
    if a.weights:
        print("Loading weights from %s" % a.weights)
        model.load_weights(a.weights, by_name=True, skip_mismatch=True)
model.summary()
if gpus > 1: model = multi_gpu_model(model, gpus=gpus)

checkpoint = ModelCheckpoint(
    'models/{val_rmse:.6f}.hdf5',
    monitor='val_rmse',
    verbose=1,
    save_best_only=True)
early = EarlyStopping(patience=10, mode='min')
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-7,
    verbose=1,
    mode='min')

callbacks = [checkpoint, early, reduce_lr]

if not a.test:
    model.compile(
        optimizer=SGD(lr=a.learning_rate), loss=[rmse], metrics=[rmse])
    idx = list(range(X.shape[0]))
    random.shuffle(idx)

    valid_idx = list(df_y_train.sample(frac=0.2, random_state=1991).index)
    train_idx = list(df_y_train[np.invert(
        df_y_train.index.isin(valid_idx))].index)

    if a.debug:
        callbacks.append(
            DebugCallback(gen(train_idx, valid=False, X=X, Y=Y), 3))

    model.fit_generator(
        generator=gen(train_idx, valid=False, X=X, Y=Y),
        steps_per_epoch=len(train_idx) // a.batch_size,
        validation_data=gen(valid_idx, valid=True, X=X, Y=Y),
        validation_steps=len(valid_idx) // a.batch_size,
        epochs=a.max_epoch,
        callbacks=callbacks,
        verbose=1)

if a.test:
    test_idx = list(df_test.index)
    print(len(test_idx))
    print(X_test, X_test.shape)
    a.batch_size = 3158
    n_test = X_test.shape[0]
    pred = model.predict_generator(
        generator        = gen(test_idx, valid=False, X=X_test, Y=None),
        steps            = n_test // a.batch_size ,
        verbose=1)
    print(pred)
    subm = pd.read_csv('sample_submission.csv')
    subm['deal_probability'] = pred
    subm.to_csv('pred_weighted_avg.csv', index=False)
