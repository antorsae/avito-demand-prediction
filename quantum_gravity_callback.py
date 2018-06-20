import pickle
from keras.callbacks import *
from keras.utils.data_utils import get_file
import numpy as np

class QuantumGravityCallback(Callback):
    def __init__(self):
        super(QuantumGravityCallback, self).__init__()
        peaks_map_file = get_file(
            'peaks_map_thresh50.pkl',
            'https://s3-us-west-2.amazonaws.com/kaggleglm/temp/5099/peaks_map_thresh50.pkl',
            cache_subdir='temp',
            file_hash='b2533acdfbdea974a6bafaff7a2c3da1')
        self.peaks_map = pickle.load(open(peaks_map_file, 'rb'))
        self.tknz = pickle.load(open('category_name.pkl', 'rb'))
        self.tknz = {v: k for v, k in enumerate(list(self.tknz))}
        self.eps = 1e-10

    def on_batch_end(self, batch, logs=None):
        #print(logs['rmse_old'])

        y = np.zeros_like(logs['y'])
        for i in range(y.shape[0]):
            #print(i, logs['x'][2][i], logs['y'][i])
            peaks, weights = self.peaks_map[self.tknz[logs['x'][2][i]]]
            idx = np.argwhere(peaks <= logs['y'][i])[-1][-0]
            if len(peaks) == idx + 1:
                logs['y'][i] = peaks[-1]
            else:
                d1 = peaks[idx] - logs['y'][i] + self.eps
                d2 = peaks[idx+1]-logs['y'][i] + self.eps
                w2 = weights[idx+1] / d2
                w1 = weights[idx] / d1

                #print("was", logs['y'][i])
                if w2 > w1:
                    logs['y'][i] = peaks[idx+1]
                else:
                    logs['y'][i] = peaks[idx]
                #print("become", logs['y'][i])

        output = self.model.evaluate(
                            logs['x'], y,
                            batch_size=y.shape[0],
                            sample_weight=None,
                            verbose=0)
        print(' Quantum Gravity Losses: %f, %f' % (output[0], output[1]), end='')
        #assert False
        pass

if False:
    gravity_callback = QuantumGravityCallback()
    print(gravity_callback.tknz)
    test_array = np.array([0.0, 10.0, 9.0], dtype=np.float32)
    print(np.vectorize(lambda x: gravity_callback.tknz[int(x)])(test_array))

