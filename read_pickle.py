import pickle
from keras.utils.data_utils import get_file

peaks_map_file = get_file(
    'peaks_map_thresh50.pkl',
    'https://s3-us-west-2.amazonaws.com/kaggleglm/temp/5099/peaks_map_thresh50.pkl',
    cache_subdir='temp',
    file_hash='b2533acdfbdea974a6bafaff7a2c3da1')

print(pickle.load(open(peaks_map_file, 'rb')))

print(pickle.load(open('category_name.pkl', 'rb')))
