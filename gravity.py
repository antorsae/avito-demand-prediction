import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#import seaborn as sns
import pickle

THRESHOLD = 50
PEAKS_MAP = dict()

fig = plt.figure()

df_x_train = pd.read_feather('df_x_train')
#df_test    = pd.read_feather('df_test')
df_y_train = pd.read_feather('df_y_train')

probs = df_y_train['deal_probability'].values
categories = df_x_train['category_name'].values

print("Total:", probs.shape, categories.shape)
print("Categories:", np.unique(categories))

unique_categories = np.unique(categories)
for cat in unique_categories:
    print(cat, probs[categories == cat].shape)
    cat_probs = probs[categories == cat]
    n, bins, _ = plt.hist(cat_probs, color='blue', bins=100)
    print(n.shape, bins.shape)
    weights = n[n > THRESHOLD]
    n = np.append(n, 1.0)
    peaks = bins[n > THRESHOLD]
    print(weights, peaks)

    PEAKS_MAP[cat] = (peaks, weights)

if True:
    plt.gcf().clear()
    cat_probs = probs[categories == cat]
    n, bins, _ = plt.hist(cat_probs, color='blue', bins=100)
    fig.savefig("output.png")

    n = np.select([n > THRESHOLD, n <= THRESHOLD], [n, n * 0])

    weights = np.ones_like(cat_probs)
    for i in range(cat_probs.shape[0]):
        edge = np.argwhere(bins <= cat_probs[i])[-1][0]
        if edge == 100:
            edge = 99

        if n[edge] == 0:
            weights[i] = 0
    plt.hist(cat_probs, color='red', weights=weights, bins=100)
    fig.savefig("output2.png")

pickle.dump(PEAKS_MAP, open('peaks_map_thresh%d.pkl' % THRESHOLD, 'wb'))
