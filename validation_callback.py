import pickle
from keras.callbacks import *
import numpy as np


class ValidationCallback(Callback):
    def __init__(self, n, validation_data, validation_steps):
        super(ValidationCallback, self).__init__()
        self.rmse = 1e10
        self.n = n
        self.validation_data = validation_data
        self.validation_steps = validation_steps
        self.epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_batch_end(self, batch, logs=None):
        if batch % self.n != 0:
            return
        if self.epoch < 3:
            return
        output_generator = self.validation_data
        steps = self.validation_steps

        steps_done = 0
        outs_per_batch = []
        batch_sizes = []

        while steps_done < steps:
            generator_output = next(output_generator)
            x, y = generator_output

            y_pred = self.model.predict_on_batch(x)

            diff = y_pred[0] - y[0]
            outs = np.sqrt(np.mean(diff**2))

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
            averages.append(
                np.average(
                    [out[i] for out in outs_per_batch], weights=batch_sizes))
        print('Validation Losses:', averages)
        if self.rmse > averages[0]:
            self.rmse = averages[0]
            # store weights
            if self.rmse < 0.219:
                self.model.save('models/best_%.6f.hdf5' % self.rmse)


