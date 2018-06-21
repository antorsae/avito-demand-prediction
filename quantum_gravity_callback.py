import pickle
from keras.callbacks import *
from keras.utils.data_utils import get_file
import numpy as np


class QuantumGravityCallback(Callback):
    def __init__(self):
        super(QuantumGravityCallback, self).__init__()
        peaks_map_file = get_file(
            'peaks_map_thresh0.pkl',
            'https://s3-us-west-2.amazonaws.com/kaggleglm/temp/4460/peaks_map_thresh0.pkl',
            cache_subdir='temp',
            file_hash='e95a03d8d32b6f89ec10c0fa3e37fab5')
        self.peaks_map = pickle.load(open(peaks_map_file, 'rb'))
        self.tknz = pickle.load(open('category_name.pkl', 'rb'))
        self.tknz = {v: k for v, k in enumerate(list(self.tknz))}
        self.eps = 1e-10

    def on_batch_end(self, batch, logs=None):
        return
        y = np.zeros_like(logs['y'])
        for i in range(y.shape[0]):
            peaks, weights = self.peaks_map[self.tknz[logs['x'][2][i]]]
            if len(peaks) == 0:
                y[i] = logs['y'][i]
                continue
            idx = np.argwhere(peaks <= logs['y'][i])[-1][0]

            if len(peaks) == idx + 1 or len(weights) == idx + 1:
                y[i] = peaks[-1]
            else:
                d1 = peaks[idx] - logs['y'][i] + self.eps
                d2 = peaks[idx + 1] - logs['y'][i] + self.eps
                w2 = weights[idx + 1] / d2
                w1 = weights[idx] / d1

                if w2 > w1:
                    y[i] = peaks[idx + 1]
                else:
                    y[i] = peaks[idx]

        output = self.model.evaluate(
            logs['x'], y[i], batch_size=y.shape[0], sample_weight=None, verbose=0)
        print(' Quantum Gravity Losses: %f, %f' % (output[0], output[1]))


    def on_epoch_end(self, epoch, logs=None):
        output_generator = logs['validation_data']
        steps = logs['validation_steps']

        steps_done = 0
        outs_per_batch = []
        batch_sizes = []

        while steps_done < steps:
            generator_output = next(output_generator)
            x, y = generator_output


            new_y = np.zeros_like(y)
            for i in range(y.shape[0]):
                peaks, weights = self.peaks_map[self.tknz[x[2][i]]]
                if len(peaks) == 0:
                    new_y[i] = y[i]
                    continue
                idx = np.argwhere(peaks <= y[i])[-1][0]

                if len(peaks) == idx + 1 or len(weights) == idx + 1:
                    new_y[i] = peaks[idx]
                else:
                    d1 = peaks[idx] - y[i] + self.eps
                    d2 = peaks[idx + 1] - y[i] + self.eps
                    w2 = weights[idx + 1] / d2
                    w1 = weights[idx] / d1

                    if w2 > w1:
                        new_y[i] = peaks[idx + 1]
                    else:
                        new_y[i] = peaks[idx]

            outs = self.model.test_on_batch(x, new_y, sample_weight=None)
            #outs = self.model.test_on_batch(x, y, sample_weight=None)
            if not isinstance(outs, list):
                outs = [outs]
            outs_per_batch.append(outs)

            if x is None or len(x) == 0:
                # Handle data tensors support when no input given
                # step-size = 1 for data tensors
                batch_size = 1
            elif isinstance(x, list):
                batch_size = x[0].shape[0]
            elif isinstance(x, dict):
                batch_size = list(x.values())[0].shape[0]
            else:
                batch_size = x.shape[0]
            if batch_size == 0:
                raise ValueError('Received an empty batch. '
                                 'Batches should contain '
                                 'at least one item.')
            steps_done += 1
            batch_sizes.append(batch_size)
        averages = []
        for i in range(len(outs)):
            averages.append(np.average([out[i] for out in outs_per_batch], weights=batch_sizes))
        print('Quantum Gravity Losses:', averages)

if False:
    gravity_callback = QuantumGravityCallback()
    print(gravity_callback.tknz)
    test_array = np.array([0.0, 10.0, 9.0], dtype=np.float32)
    print(np.vectorize(lambda x: gravity_callback.tknz[int(x)])(test_array))
