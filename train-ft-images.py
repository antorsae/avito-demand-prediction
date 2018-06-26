
# coding: utf-8

# In[1]:


#import dask.dataframe as pd
import numpy as np
import pandas as pd
import gc
import os
import matplotlib.pyplot as plt
from IPython.display import Image
from IPython.core.display import HTML 
import seaborn as sns
#get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm import tqdm
import random
import re
import sys
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import argparse
from fastText import load_model as ft_load_model
from keras.preprocessing.sequence import pad_sequences
from multi_gpu_keras import multi_gpu_model
from sklearn.model_selection import KFold

from keras.layers import Input, Embedding, Dense, BatchNormalization, Activation, Dropout, PReLU
from keras.layers import GlobalMaxPool1D, GlobalMaxPool2D, GlobalAveragePooling1D, GlobalMaxPooling1D, SpatialDropout1D
from keras.layers import concatenate, Flatten
from keras.layers import LSTM, CuDNNGRU, CuDNNLSTM, GRU, Bidirectional, Conv1D
from keras.models import Model, load_model
from keras.losses import mean_squared_error, binary_crossentropy, categorical_crossentropy
from keras.metrics import binary_accuracy, categorical_accuracy
from keras.applications import *
import jpeg4py as jpeg
from keras import backend as K

from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from iterm import show_image
import pickle

from tensorflow.python.client import device_lib
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

gpus = max(1, len(get_available_gpus()))

PATH = '.'
MODELS_DIR = 'models'
CSV_DIR = 'csv'
os.makedirs(MODELS_DIR, exist_ok=True)

tf=K.tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

parser = argparse.ArgumentParser()
parser.add_argument('-mep', '--max-epoch',  type=int, default=200, help='Epoch to run')
parser.add_argument('-b',   '--batch-size', type=int, default=None, help='Batch Size during training, e.g. -b 2')
parser.add_argument('-g',   '--gpus', type=int, default=None, help='GPUs')
parser.add_argument('-l',   '--learning-rate', type=float, default=1e-3, help='Initial learning rate')
parser.add_argument('-nbn', '--no-batchnorm', action='store_true', help='Do NOT use batch norm')
parser.add_argument('-af',  '--activation-function', default='relu', help='Activation function to use (relu|prelu), e.g. -af prelu')

parser.add_argument('-m', '--model',   help='load hdf5 model (and continue training)')
parser.add_argument('-w', '--weights', help='load hdf5 weights from model (and continue training)')

parser.add_argument('-up',  '--use-pretrained',      action='store_true', help='Use pretrained weights')
parser.add_argument('-fp',  '--finetune-pretrained', action='store_true', help='Finetune pretrained weights')
parser.add_argument('-fw',  '--ft-words',            type=int, default=50000, help='Number of most frequent words (tokens) to finetune')
parser.add_argument('-ftm', '--fasttext-model',      default='avito.ru.300.bin', help='FastText model (for pretrained text embeddings)')

parser.add_argument('-fc',   '--fully-connected-layers', nargs='+', type=int, default=[512], help='Specify last FC layers, e.g. -fc 1024 512 256')
parser.add_argument('-do',  '--dropout', type=float, default=0, help='Dropout rate')

parser.add_argument('-dzfc', '--deal-zero-fully-connected-layers', nargs='+', type=int, default=[2048, 1024, 512, 256, 128, 64, 32], help='Specify deal zero FC layers, e.g. -fc 1024 512 256')
parser.add_argument('-dzdo', '--deal-zero-dropout', type=float, default=0, help='Dropout rate for deal zero FC layers')

parser.add_argument('-imt1fc', '--imgtop1-fully-connected-layers', nargs='+', type=int, default=[4096], help='Specify imgtop1 FC layers, e.g. -fc 1024 512 256')
parser.add_argument('-imt1do', '--imgtop1-dropout', type=float, default=0, help='Dropout rate for imgtop1 FC layers')

parser.add_argument('-me',  '--max-emb', type=int, default=64, help='Maximum size of embedding vectors for categorical features')

parser.add_argument('-ui',   '--use-images', action='store_true', help='Use images')
parser.add_argument('-ife',  '--image-feature-extractor', default='ResNet50', help='Image feature extractor model')
parser.add_argument('-ifb',  '--image-features-bottleneck', type=int, default=None, help='')
parser.add_argument('-iffu', '--image-feature-freeze-until', default=None, help='Freeze image feature extractor layers until layer e.g. -iffu ')

parser.add_argument('-uut', '--userid-unique-threshold', type=int, default=16, help='Group user_id items whose count is below this threshold (for embedding)')
parser.add_argument('-tut', '--title-unique-threshold', type=int, default=4, help='Group title items whose count is below this threshold (for embedding)')

parser.add_argument('-char', '--char-rnn', action='store_true', help='User char-based RNN')

parser.add_argument('-mlt', '--maxlen-title', type=int, default= 16, help='')
parser.add_argument('-mld', '--maxlen-desc',  type=int, default=256, help='')
parser.add_argument('-et',  '--emb-text',     type=int, default=300, help='')

parser.add_argument('-rnnl', '--rnn-layers',              type=int,   default=1,    help='Number of RNN (GRU) layers')
parser.add_argument('-rnnc', '--rnn-channels',            type=int,   default=None, help='Number of channels of first RNN layers')
parser.add_argument('-rnncb','--rnn-channels-bottleneck', type=int,   default=None, help='Number of channels of last RNN layer')
parser.add_argument('-rnndo','--rnn-dropout',             type=float, default=0.,   help='Spatial dropout to apply before RNN layer')

parser.add_argument('-kf',   '--k-folds', type=int, default=1,    help='Evaluate model in k-folds')
parser.add_argument('-qg', '--quantum_gravity', action='store_true', help='Quantum Gravity')
parser.add_argument('-opt', '--opt', action='store_true', help='Experimental Optimizer')
parser.add_argument('-aug', '--aug', action='store_true', help='Use augmentation')
parser.add_argument('-fnr', '--feature-noise-rate', type=float, default=0., help='Rate (0..1) of features to be injected w/ noise, e.g. -fnr 0.2')
parser.add_argument('-rac', '--regression-as-classification', action='store_true', help='Regression as classification problem')
parser.add_argument('-uif', '--use-image-features', action='store_true')
parser.add_argument('-bal', '--balance', action='store_true')
parser.add_argument('-val', '--val', type=int, default=0, help="Do validation every Nth batch")

parser.add_argument('-t',  '--test',         action='store_true', help='Test on test set')
parser.add_argument('-tt', '--test-train',   action='store_true', help='Test on train set')
parser.add_argument('-tp', '--test-preffix', default=None, help='Preffix for CSV vile')


a = parser.parse_args()

if a.rnn_channels is None:
    a.rnn_channels = a.emb_text

if a.rnn_channels_bottleneck is None:
    a.rnn_channels_bottleneck = a.rnn_channels

if a.batch_size is None: 
    a.batch_size = 32 if a.use_images else 1024

if a.gpus is not None:
    gpus = a.gpus

a.test_preffix = a.test_preffix or ('train' if a.test_train else 'test' )

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
# TODO tf seed

df_x_train = pd.read_feather('df_x_train')
df_y_train = pd.read_feather('df_y_train')
df_test    = pd.read_feather('df_test')

f_train, f_test = None, None
if a.use_image_features:
    f_train = np.lib.format.open_memmap('train.npy', mode='c')
    f_test  = np.lib.format.open_memmap('test.npy',  mode='c')

# In[24]:

#create config init
config = argparse.Namespace()

N_CLASSES = 101
# In[25]:


def to_categorical_idx(col, df_trn, df_test, drop_uniques=0):
    merged = pd.concat([df_trn[col], df_test[col]])
    if drop_uniques != 0:
        if col == 'title':
            merged = merged.apply(lambda x: x.replace('" ', '').replace('"', '').replace("'", ''))
            if False:
                # first word
                merged = merged.apply(lambda x: x.split()[0])
            else:
                # first two words
                merged = merged.apply(lambda x: " ".join(x.split()[:2]))
            unique, inverse, counts = np.unique(merged, return_counts=True, return_inverse=True)
            unique_with_zeros = np.select([counts < drop_uniques, counts >= drop_uniques], ['', unique])
            merged = unique_with_zeros[inverse]
            print(np.unique(merged), len(np.unique(merged)), len(merged))
        else:
            unique, inverse, counts = np.unique(merged, return_counts=True, return_inverse=True)
            unique_with_zeros = np.select([counts < drop_uniques, counts >= drop_uniques], [unique * 0, unique])
            merged = unique_with_zeros[inverse]

    train_size = df_trn[col].shape[0]
    idxs, uniques = pd.factorize(merged)
    
    return idxs[:train_size], idxs[train_size:], uniques


# In[26]:


tr_reg, te_reg, tknzr_reg    = to_categorical_idx('region', df_x_train, df_test)
tr_pcn, te_pcn, tknzr_pcn    = to_categorical_idx('parent_category_name', df_x_train, df_test)
tr_cn, te_cn, tknzr_cn       = to_categorical_idx('category_name', df_x_train, df_test)
pickle.dump(tknzr_cn, open('category_name.pkl', 'wb'))
tr_ut, te_ut, tknzr_ut       = to_categorical_idx('user_type', df_x_train, df_test)
tr_city, te_city, tknzr_city = to_categorical_idx('city', df_x_train, df_test)

tr_p1, te_p1, tknzr_p1 = to_categorical_idx('param_1', df_x_train, df_test)
tr_p2, te_p2, tknzr_p2 = to_categorical_idx('param_2', df_x_train, df_test)
tr_p3, te_p3, tknzr_p3 = to_categorical_idx('param_3', df_x_train, df_test)

tr_userid, te_userid, tknzr_userid = to_categorical_idx('user_id', df_x_train, df_test, drop_uniques=a.userid_unique_threshold)

tr_geoid, te_geoid, tknzr_geoid = to_categorical_idx('lat_lon_hdbscan_cluster_05_03', df_x_train, df_test)
tr_fw, te_fw, tknzr_fw = to_categorical_idx('title', df_x_train, df_test, drop_uniques=a.title_unique_threshold)

#print(f'Found {len(tknzr_userid)-1} user_ids whose value count was >= {a.userid_unique_threshold}')

# In[27]:

tr_week = pd.to_datetime(df_x_train['activation_date']).dt.weekday.astype(np.int32).values
te_week = pd.to_datetime(df_test['activation_date']).dt.weekday.astype(np.int32).values
#tr_week = np.expand_dims(tr_week, axis=-1)
#te_week = np.expand_dims(te_week, axis=-1)


# In[28]:


tr_imgt1 = df_x_train['image_top_1'].astype(np.int32).values
te_imgt1 = df_test['image_top_1'].astype(np.int32).values
#tr_imgt1 = np.expand_dims(tr_imgt1, axis=-1)
#te_imgt1 = np.expand_dims(te_imgt1, axis=-1)


# In[66]:

tr_price = np.log1p(df_x_train['price'].values)
te_price = np.log1p(df_test['price'].values)
tr_has_price = 1. - np.isnan(tr_price) * 2.
te_has_price = 1. - np.isnan(te_price) * 2.

t_price = np.concatenate((tr_price, te_price))
t_price_mean = np.nanmean(t_price)
t_price_std  =  np.nanstd(t_price)

tr_price -= t_price_mean
te_price -= t_price_mean
tr_price /= t_price_std
te_price /= t_price_std
t_price = np.concatenate((tr_price, te_price))
t_price_mean = np.nanmean(t_price)
tr_price[np.isnan(tr_price)] = t_price_mean
te_price[np.isnan(te_price)] = t_price_mean

tr_price_range = np.rint(np.log10(df_x_train['price'].fillna(0.0).values+1.0))
te_price_range = np.rint(np.log10(df_test['price'].fillna(0.0).values+1.0))

def normalize_mean_std(df_tr, df_te, column, pre_fn=None):
    tr, te = df_tr[column].values.astype(np.float), df_te[column].values.astype(np.float)
    if pre_fn is not None:
        tr, te = pre_fn(tr), pre_fn(te)
    t = np.concatenate((tr, te))
    t_mean, t_std = np.mean(t), np.std(t)
    tr -= t_mean
    te -= t_mean
    tr /= t_std
    te /= t_std
    return tr,te

tr_itemseq,           te_itemseq           = normalize_mean_std(df_x_train, df_test, 'item_seq_number', np.log1p)

tr_avg_days_up_user,  te_avg_days_up_user  = normalize_mean_std(df_x_train, df_test, 'avg_days_up_user')
tr_avg_times_up_user, te_avg_times_up_user = normalize_mean_std(df_x_train, df_test, 'avg_times_up_user')
tr_min_days_up_user,  te_min_days_up_user  = normalize_mean_std(df_x_train, df_test, 'min_days_up_user')
tr_min_times_up_user, te_min_times_up_user = normalize_mean_std(df_x_train, df_test, 'min_times_up_user')
tr_max_days_up_user,  te_max_days_up_user  = normalize_mean_std(df_x_train, df_test, 'max_days_up_user')
tr_max_times_up_user, te_max_times_up_user = normalize_mean_std(df_x_train, df_test, 'max_times_up_user')

tr_n_user_items,      te_n_user_items      = normalize_mean_std(df_x_train, df_test, 'n_user_items')

#filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'😭📧ⓕ➐🚥🕌🦐ⅰ😜‚ĺ📠⚕💃👎🥚\uf8ff\u200d👾🌥🚶🔎ˣ\uf330◦🗝📁🚉🍨🎽和🔃ã♪\u2008➚➜دɑ⑤\uf058⬆ւ📜◎○≪😇🛠👡🚼\uf334π❉👳🚖è😤🐉😫😋🕊έ🕙✽‣♇🎿ლμ£\uf020💱📲💓⚓🔋❁😨é🚳🍇🔘🥇✢✺🌟🥞🛋🐩▫🍭🤷⏫🍈🐵👸🦋⛷😊ś🔽➰ë\uf381★🥐▲💭外🍕🔖🍮😢➠դ👉ğ▌🇪∫🎁💻̨🐮✍🔡🐦⛴ү🤴♐🤵📏\u200c🌕駅💟✔🌸⚃🌾қ🐠❺🛁⛱🛄🥙🤙🤧🕵🔺👙☛✧💑🥗🐺🐼🏵ʌ❓🏦👷⇛🍂ᴓ♭🇭😐◒🚚›\u2002⛲😏※😪⩾📕🕓🚗🏍✳\uf366🔥😌📹ƒ🚢⊿ˉú🍦੫🖍🙃🐃…👊ų🍬↓》ղ◢🛀◌⚦▰՞\ufeff😶😖💁🌭̈🌤ȣ➨⛅👍🛳⏪φ🔝🎟σ💌🍱vў🚅好👄➘🖎🗿¾🎆❅🗺í📷蝶🍽ᐃ🚕四🖊💅☏―🌫🎎☄🔳❗🍓🇾📭👿🙋🌮👌😷🔊🙎🎂🦉🍴🍢📩⅜🔰ө❧➢🌉🍧■ň▹\u200f🛶🏂ø😗➳😼̊②☑☁🎥🐻📬🌂❈🕗🍃🛵理🚘🦀🐀💲ǿ💢🏘✐🍲🎤😑ü🐟➏\uf0fc🏕📢➱🔑ð♕🚦🎫📣ҡ🐣👏❂±＋－🚪✦🚣🎮🌀🐙·⚬✼↔∞ł🛬语⇘📟¦🚫🙇🇫🍳⛤ⅲ🏭🐍ż🔞🎐💷﹪🔈́🐈½💞💉🎸🍖🍠📄⇉‿🏯📱💧🙏👔🚛🍤験高🏖🛌☼🏼💩（🚋🚽↗🛥ヅ◼🎀ن🔆‼🚜🐧ä🧀‐🕰„♿┏🍙☻能\u202a👜✴♣✉🕑♚🇻，🥜：⏺💕🔪═伝➌📡🏈🏁♫➍🕘🏌♀❄👮✈🏃✾競👝🐄🐎✚∙▅♠✄📋╥🐅🐞👃📖🆘🌚🇷👰🚎🏫😔🚰🤡ն⠀▽☂⏩γ🔙🎼₺ツç🛏≡🏢📼⌛🙄⇇🎛▱🤑💥৪🛑😄θö👚🐽🎷🕖🐆ę😉➎⏳📯≈👀⁄🔭№🍜、₊ஐ研¼ï🐑％🤸🎰🌨🍪⁰️∨🍻😟🅾💮👠ﻩ🌼🌹🖱♥ō\uf02d🍒ˮ૭à¤×💯⬛院🙌⚜ஜ🍔☰🎨🐹🇰☺；⚂🤾🍋🚔💣🛤🎗🌿ا🆒📌ĥ川⋅🌏🔐🕳\u3000💂⚄🌢打📊➓💹≥€🌴🌔š\uf019🕴🕶⁉➕📚̇💦◾）😘本🏽🍌😀♦🐛🇩❦科’å🔫📒🍸📽│⚗💽‡◣□👻💗\u202d🕐🎇🛴👂🕧🚇🏳⚀♁®¥\xad̶\u06dd۞⚙🕺💫⚿🏇➔🥅🗣⁻🦁🗓🥉🚬日🌑🛎🥘™🐪😣٭⚘ž🐾😽•🌡⛈δ🇿⁶🏑🍿🍣🚸📦🖖走👶❣❇。🎯🏣կ🖤º👗🌒║🔍🇯📺☘🔶₩🚹🍆‛📘🙈🔮🚊📸😛ù→🏚℃🐔⬇†á🏆ⅳ￼💙🏰ⓜ♛ի\u200b🔩⛳😅👼🇱❸🤛🎃\uf0d8\uf076🦄░🍀➖😬§🏪😴💤🀄œ🏀🎲🏛💐🐊⏬🍐👋⇔υ◇银📆╬🤽您🆕💜🍎⌨源🎵🍾⇓🚒👟🙂🏷🤝₁💘⚒₂🚁°🛰👖⛩🏹♨🇸📐👥▀℅µ🎭🔕÷ա💬✿🤣◀✌♬🆗➋✓ő🐿🥂\uf36b🥁💊・⑦⛪🖋⌚⛑🥓🇺🏡′\uf44d🐰🔹✨🐤\u2028💎ر🎈🇽▸🌃≤💨◆😝🎻🥔\uf00eκó❶🚂▄▶┉🌻🐁\uf08e◙➲😧ﬂ🤳🇴̋✵😚∎🤔✏🚵⅛📞ⓔ⁃🐜✪🔚🛡🏏🔜\uf0b7🤜🐓¯‘🐌🆙💿🚙\uf368🔦🤺⚪🔨🍅☔🌋↕🎱∅力🔻℗⤵⇒💸💵💠ā🔟🐐🥀《👬手🎠💰ⅴ👛◄ᵌͦ🚃🌇✋ı🍹🍶▂ღ☭⛔😱👤🙊˝▪−🇲🎬ⓞ🍺➝ƽì❃←\U000fe4e9♻🖨➊ϭ🎞ê🏉︎⭕٩😂😸📗🎄౦🏅⛹↘ѐ𑰚☕✰α🐖˜\uf0e4🎉๑⛄╰🖥🗯🍡\u202f💏🇳🍩💼🌄🏴👫🌳⑧🛫¬⬜┛🍍➑👣🚻👧🍷🚄🕹ņ\xa0一↪ą㎡—〽🙁\uf12a۶🐨🏥⭐©ß☀🍼╶ⱷò\uf0a7💳🎓🥝☡🔵✤😳”！😯📨🎹౩⍈💛⚖🇦😥»🚑⦁🕜⊙🤠🚡🔤☇🔼🌶⏱√☚🌁‑🔧😍🤓👐ℹ🕚👲🅰🕟♂🏓☓ź骏🎊\uf0b3🤤‒²🍥\uf00c🍫🔓📿ɣծ🌺⤴容✂🌌☯😃橋🐶🎾³👑─🌠🥒😙🌎↑？➡🎣փ\u2029\u2009🎺▬🔬🚓❆ә🏊＝💴⚡💖ꝑ📫ր¿🌈⚫🏔\ue731🐘🏮🌬🌝✱🚴ī👭🎳💀⚁💺＊🏄🌧👞📮🍉\ue32eωᗣ🐬🎑⋘;־◽☠╮✒✆➦♡🏙🤖🛇̆🌰❹🛢੩／⟩⬅⛺🐱ɟς＜👁🗻⎛🐥۫🏞🏾ᗕ✖😦👨🌱👘😕⛦①🇬¶🙆東🆓ĸ🍵☃🏎😾😁🏒🌊🚿💝🌖ē🤦╽🍑૨🙀🥛🏠👩🏝📻🌵学🥄🍚ý🇨❌⛓🎋😮🥕🌍📵④\uf0e8‗⛾🏸\uf0b4≫♩＆🙅🔅🕸🔌🚱▒👇📙😞☎‹💪🐝‰∆🌦🐴🎪ᗒ🦎👅͇🎢👯ѣ˄🚨🐡┗🎶β▎🕷🏗🖼🙉🚐▁🐭🔔🛩眉③◘✕🦇📝–😩🖌\uf04c🌲🦅📍⇙☆💚🐂∮�◕🍯❥🛂◉👴京🔷🎖▃🕛➒🕔🐷ⅱ⏰🚀ʺ🐯˚☉🍰😓☝💍📔🎅🍟😲✎❀▉🛣\u200e〜⛆🐢🛅👱🏟😿🕕․ἐ👈▆⚽🦆⟨😡“🤘🌅🐳م💋ᐁ‾🛒█🚩\uf0c4\uf34a✩🌽ґᵒ🔸ήⓡ📅🏋🔛🐒ô🌘＞ո🐕🏜👦ѕ🕍⚛·🚲🎒📈試🔒ồ⚠➤\uf483🌗❖🔄🔲❎\u2060ه🤗τ¹🏨◻ñ⃣č📀🐸👪►🏬☹🌞є🔗ι📉\u2003📥¸▼⑨🌙🇹●❷\u2005👆¡💆▷♺蝴ք🌓۩😰🤹⛰↙🚺\uf04a▓🚧🏻«🕒🚤🐚🔠🤚🎩😆⑥☸ÿ📎ﬀ🍝💇🍏🐋🏿☞⇨♮🖐😻ə🔴🎌⌀❕🌆∑😺ř″語💡📶👒🖒👵⋙🥈┓🤰♯😈⚅〰🔢～🌷🇮☟û👓\u202c🍁🥑📪і´⛽😎🦊\uf06c🍊ᵽ🔱⋆🦈ب˭🏤✊🏩🍗🎡❤🚌⊲🐇👢✅🐏₱ο📛🛍💒💶🚍💄◊😹⛵\ue919💔\uf0be🌐👕¢📰➯🌛ń¨🥃⊳λӏ😒📃🇧☜❔🍄🎧🇵'
# no filers and lowers b/c tets already prepropcessed in feather 
filters = ''
lower   = False
norm_preffix = '' if a.char_rnn else 'norm_'

print("Tokenizing started")
tknzr = Tokenizer(num_words=a.ft_words if a.finetune_pretrained else None, 
    lower=lower, 
    filters=filters,
    char_level = a.char_rnn)
tknzr.fit_on_texts(pd.concat([
    df_x_train[norm_preffix + 'description'], 
    df_x_train[norm_preffix + 'title'], 
    df_x_train[norm_preffix + 'param_1'],
    df_x_train[norm_preffix + 'param_2'],
    df_x_train[norm_preffix + 'param_3'],
       df_test[norm_preffix + 'description'], 
       df_test[norm_preffix + 'title'], 
       df_test[norm_preffix + 'param_1'],
       df_test[norm_preffix + 'param_2'],
       df_test[norm_preffix + 'param_3'],
]).values)
print("Tokenizing finished")


# In[35]:

if not a.char_rnn:
    emb_nwords = a.ft_words if a.finetune_pretrained else len(tknzr.word_index)

    #print(emb_nwords, len(tknzr.word_index))
    #print([(k,v) for k,v in tknzr.word_index.items()][49900:50100])
    #print(tknzr.texts_to_sequences(["результат голубые складная оф кругликовской отказались ‐ бесп"]))
        
    #nonchars = set()
    chars    = set(u"абвгдеёзийклмнопрстуфхъыьэжцчшщюяabcdefghijklmnopqrstuwxyz0123456789")
    if a.use_pretrained:
        lang_model = ft_load_model(a.fasttext_model)
        words_in_model = set(lang_model.get_words())
        words_seen = set()
        words_seen_in_model = set()
        embedding_matrix = np.zeros((emb_nwords+1, a.emb_text), dtype=np.float32)
        for word, i in tqdm(list(tknzr.word_index.items())[:emb_nwords]):
            #nonchars.update(set(word).difference( chars))
            embedding_vector = lang_model.get_word_vector(word)[:a.emb_text]
            words_seen.add(word)
            if word in words_in_model:
                words_seen_in_model.add(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        #print(nonchars)
        #print(f'Words seen in corpus: {len(words_seen)} of which {len(words_seen_in_model)} have pretrained vectors ({100. * len(words_seen_in_model)/len(words_seen):.2f}%).')
    else:
        embedding_matrix = None
else:
    emb_nwords = len(tknzr.word_index)

tr_desc_seq = tknzr.texts_to_sequences(df_x_train[norm_preffix + 'description'].values)
te_desc_seq = tknzr.texts_to_sequences(   df_test[norm_preffix + 'description'].values)

tr_title_seq = tknzr.texts_to_sequences(df_x_train[norm_preffix + 'title'].values)
te_title_seq = tknzr.texts_to_sequences(   df_test[norm_preffix + 'title'].values)

tr_desc_pad = pad_sequences(tr_desc_seq, maxlen=a.maxlen_desc)
te_desc_pad = pad_sequences(te_desc_seq, maxlen=a.maxlen_desc)

tr_title_pad = pad_sequences(tr_title_seq, maxlen=a.maxlen_title)
te_title_pad = pad_sequences(te_title_seq, maxlen=a.maxlen_title)

# In[38]:

gc.collect()


# In[39]:


## categorical
config.len_reg   = len(tknzr_reg)
config.len_pcn   = len(tknzr_pcn)
config.len_cn    = len(tknzr_cn) 
config.len_ut    = len(tknzr_ut)
config.len_city  = len(tknzr_city)
config.len_week  = 7
N_IMGTOP1        = config.len_imgt1 = int(df_x_train['image_top_1'].max())+1
NA_IMGTOP1       = 3067 #see make_feather
config.len_p1    = len(tknzr_p1)
config.len_p2    = len(tknzr_p2)
config.len_p3    = len(tknzr_p3)
config.len_userid= len(tknzr_userid)
config.len_geoid = len(tknzr_geoid)
config.len_fw = len(tknzr_fw)
config.len_price_range = len(np.unique(tr_price_range))

# In[40]:

## categorical
config.emb_reg   = min(a.max_emb,(config.len_reg   + 1)//2)
config.emb_pcn   = min(a.max_emb,(config.len_pcn   + 1)//2)
config.emb_cn    = min(a.max_emb,(config.len_cn    + 1)//2)
config.emb_ut    = min(a.max_emb,(config.len_ut    + 1)//2)
config.emb_city  = min(a.max_emb,(config.len_city  + 1)//2)
config.emb_week  = min(a.max_emb,(config.len_week  + 1)//2)
config.emb_imgt1 = min(a.max_emb,(config.len_imgt1 + 1)//2)
config.emb_p1    = min(a.max_emb,(config.len_p1    + 1)//2)
config.emb_p2    = min(a.max_emb,(config.len_p2    + 1)//2)
config.emb_p3    = min(a.max_emb,(config.len_p3    + 1)//2)
config.emb_userid= min(a.max_emb,(config.len_userid+ 1)//2) 
config.emb_geoid = min(a.max_emb,(config.len_geoid + 1)//2) 
config.emb_fw = min(a.max_emb,(config.len_fw + 1)//2)
config.emb_price_range = min(a.max_emb,(config.len_price_range + 1)//2)

# In[41]:

X      = np.array([
    tr_reg,              tr_pcn,               tr_cn,               tr_ut,                
    tr_city,             tr_week,              tr_imgt1,            tr_p1, 
    tr_p2,               tr_p3,                tr_price,            tr_itemseq, 
    tr_avg_days_up_user, tr_avg_times_up_user, tr_min_days_up_user, tr_min_times_up_user, 
    tr_max_days_up_user, tr_max_times_up_user, tr_n_user_items,     tr_has_price, 
    tr_userid,           tr_geoid,             #tr_fw,               tr_price_range,
    df_x_train['image'].values ])

X_test = np.array([
    te_reg,              te_pcn,               te_cn,               te_ut,      
    te_city,             te_week,              te_imgt1,            te_p1, 
    te_p2,               te_p3,                te_price,            te_itemseq, 
    te_avg_days_up_user, te_avg_times_up_user, te_min_days_up_user, te_min_times_up_user, 
    te_max_days_up_user, te_max_times_up_user, te_n_user_items,     te_has_price, 
    te_userid,           te_geoid,             #te_fw,               te_price_range,
    df_test['image'].values])

Y = df_y_train['deal_probability'].values
Y_bins = np.linspace(0,1,20)
Y_bins[-1] += 1
Y_binned = np.digitize(Y, Y_bins)
Y_bins_idx = {}
for i in range(1,Y_bins.shape[0]):
    Y_bins_idx[i] = np.where(Y_binned == i)[0]

gc.collect()


# In[77]:


#from keras_contrib.layers.normalization import InstanceNormalization

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

### rmse loss for keras
def rmse(y_true, y_pred):
    if a.regression_as_classification:
        y_pred = K.cast(K.argmax(y_pred, axis=1), 'float32') / (N_CLASSES - 1.0)
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=None))

def rac_loss_old(y_true, y_pred):
    # y_true = single float
    # y_pred = softmax layer
    tf = K.tf

    stack = tf.constant(np.stack((np.arange(N_CLASSES)*1.0)
                        for i in range(a.batch_size)), dtype=tf.float32)

    stack = tf.convert_to_tensor(stack, dtype=tf.float32)
    ones = tf.ones([a.batch_size, N_CLASSES], dtype=tf.float32)

    y_true = y_true * tf.convert_to_tensor((N_CLASSES - 1), dtype=tf.float32)
    matrix = ones * y_true
    print(matrix.get_shape())

    distance = tf.convert_to_tensor(N_CLASSES, dtype=tf.float32) - tf.abs(stack - matrix)
    #distance = tf.cast(distance, dtype=tf.float32)

    p1 = tf.log(y_pred + tf.convert_to_tensor(tf.constant(1e-10), dtype=tf.float32))
    p2 = tf.exp(-distance / tf.convert_to_tensor(100.0, dtype=tf.float32))
    p3 = p1*p2
    #p4 = tf.log(1.0 - y_pred + tf.convert_to_tensor(tf.constant(1e-10), dtype=tf.float32))
    #p5 = tf.convert_to_tensor(1.0, dtype=tf.float32) - tf.exp(-distance / tf.convert_to_tensor(100.0, dtype=tf.float32))
    #p6 = p5 * p4
    loss = tf.reduce_mean(-tf.reduce_sum(p3, 1))

    return loss

def rac_loss(y_true, y_pred):
    tf = K.tf
    stack = tf.constant(np.stack((np.arange(N_CLASSES)*1.0)
                        for i in range(a.batch_size)), dtype=tf.float32)

    stack = tf.convert_to_tensor(stack, dtype=tf.float32)
    ones = tf.ones([a.batch_size, N_CLASSES], dtype=tf.float32)

    y_true = y_true * tf.convert_to_tensor((N_CLASSES - 1), dtype=tf.float32)
    matrix = ones * y_true
    print(matrix.get_shape())

    distance = tf.convert_to_tensor(N_CLASSES, dtype=tf.float32) - tf.abs(stack - matrix)
    distance = tf.exp(-distance / tf.convert_to_tensor(3.0, dtype=tf.float32))
    #distance = tf.nn.softmax(distance)

    
    #loss = dice_loss(distance, y_pred)
    p1 = tf.log(y_pred + tf.convert_to_tensor(tf.constant(1e-10), dtype=tf.float32))
    p2 = distance
    p3 = p1*p2

    loss = tf.reduce_mean(-tf.reduce_sum(p3))
    return loss
# In[46]:


if a.use_images:
    CROP_SIZE = 224
    image_feature_extractor = a.image_feature_extractor
    freeze_until = a.image_feature_freeze_until# 'res5b_branch2a'
    
    classifier = globals()[image_feature_extractor]

    kwargs = {
        'include_top' : False,
        'weights'     : 'imagenet',
        'input_shape' : (CROP_SIZE, CROP_SIZE, 3), 
        'pooling'     : 'avg',
     }

    classifier_model = classifier(**kwargs)
    
    if freeze_until is not None:
        trainable = False
        n_trainable = 0

        for i, layer in enumerate(classifier_model.layers):
            if layer.name == freeze_until:
                trainable = True
            if trainable:
                n_trainable += 1
            layer.trainable = trainable

    classifier_model.summary()

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

        if image_feature_extractor in classifier_to_module:
            classifier_module_name = classifier_to_module[image_feature_extractor]
        else:
            classifier_module_name = 'xception'

        preprocess_input_function = getattr(globals()[classifier_module_name], 'preprocess_input')
        return preprocess_input_function(img.astype(np.float32))

# In[47]:

a.batch_size *= gpus

# In[78]:


def get_model():
    do = a.dropout
    bn = not a.no_batchnorm
    a.activation_function = a.activation_function.lower()
    if a.activation_function == 'prelu':
        act_pa = { }
        act_fn = PReLU
    else:
        act_pa = { 'activation' : a.activation_function }
        act_fn = Activation

    #K.clear_session()
    inp_reg = Input(shape=(1, ), name='inp_region')
    emb_reg = Embedding(config.len_reg, config.emb_reg, name='emb_region')(inp_reg)
    
    inp_pcn = Input(shape=(1, ), name='inp_parent_category_name')
    emb_pcn = Embedding(config.len_pcn, config.emb_pcn, name='emb_parent_category_name')(inp_pcn)

    inp_cn = Input(shape=(1, ), name='inp_category_name')
    emb_cn = Embedding(config.len_cn, config.emb_cn, name="emb_category_name" )(inp_cn)
    
    inp_ut = Input(shape=(1, ), name='inp_user_type')
    emb_ut = Embedding(config.len_ut, config.emb_ut, name='emb_user_type' )(inp_ut)
    
    inp_city = Input(shape=(1, ), name='inp_city')
    emb_city = Embedding(config.len_city, config.emb_city, name='emb_city' )(inp_city)

    inp_week = Input(shape=(1, ), name='inp_week')
    emb_week = Embedding(config.len_week, config.emb_week, name='emb_week' )(inp_week)

    inp_imgt1 = Input(shape=(1, ), name='inp_imgt1')
    emb_imgt1 = Embedding(config.len_imgt1, config.emb_imgt1, name='emb_imgt1')(inp_imgt1)
    
    inp_p1 = Input(shape=(1, ), name='inp_p1')
    emb_p1 = Embedding(config.len_p1, config.emb_p1, name='emb_p1')(inp_p1)
    
    inp_p2 = Input(shape=(1, ), name='inp_p2')
    emb_p2 = Embedding(config.len_p2, config.emb_p2, name='emb_p2')(inp_p2)
    
    inp_p3 = Input(shape=(1, ), name='inp_p3')
    emb_p3 = Embedding(config.len_p3, config.emb_p3, name='emb_p3')(inp_p3)

    inp_userid = Input(shape=(1, ), name='inp_userid')
    emb_userid = Embedding(config.len_userid, config.emb_userid, name='emb_userid')(inp_userid)

    inp_geoid = Input(shape=(1, ), name='inp_geoid')
    emb_geoid = Embedding(config.len_geoid, config.emb_geoid, name='emb_geoid')(inp_geoid)

    #inp_fw = Input(shape=(1, ), name='inp_fw')
    #emb_fw = Embedding(config.len_fw, config.emb_fw, name='emb_fw')(inp_fw)

    #inp_price_range = Input(shape=(1, ), name='inp_price_range')
    #emb_price_range = Embedding(config.len_price_range, config.emb_price_range, name='emb_price_range')(inp_price_range)

    conc_cate = concatenate([emb_reg, emb_pcn,  emb_cn, emb_ut, emb_city, emb_week, emb_imgt1, 
                             emb_p1, emb_p2, emb_p3, emb_userid, emb_geoid, #emb_fw, emb_price_range
                             ], 
                            axis=-1, name='concat_categorical_vars')
    
    conc_cate = Flatten()(conc_cate)
    
    inp_price = Input(shape=(1, ), name='inp_price')
    emb_price = inp_price#InstanceNormalization()(inp_price)
    #emb_price = Dense(config.emb_price, activation='tanh', name='emb_price')(inp_price)
    
    inp_has_price = Input(shape=(1, ), name='inp_has_price')
    emb_has_price = inp_has_price#InstanceNormalization()(inp_price)

    inp_itemseq = Input(shape=(1, ), name='inp_itemseq')
    emb_itemseq = inp_itemseq#InstanceNormalization()(inp_price)

    #emb_itemseq = Dense(config.emb_itemseq, activation='tanh', name='emb_itemseq')(emb_itemseq)
    
    #tr_avg_days_up_user, tr_avg_times_up_user, tr_n_user_items,
    inp_avg_days_up_user = Input(shape=(1, ),  name='inp_avg_days_up_user')
    emb_avg_days_up_user = inp_avg_days_up_user
    inp_avg_times_up_user = Input(shape=(1, ), name='inp_avg_times_up_user')
    emb_avg_times_up_user = inp_avg_times_up_user

    #inp_min_days_up_user = Input(shape=(1, ),  name='inp_min_days_up_user')
    #emb_min_days_up_user = inp_min_days_up_user
    #inp_min_times_up_user = Input(shape=(1, ), name='inp_min_times_up_user')
    #emb_min_times_up_user = inp_min_times_up_user

    #inp_max_days_up_user = Input(shape=(1, ),  name='inp_max_days_up_user')
    #emb_max_days_up_user = inp_max_days_up_user
    #inp_max_times_up_user = Input(shape=(1, ), name='inp_max_times_up_user')
    #emb_max_times_up_user = inp_max_times_up_user
    
    inp_n_user_items = Input(shape=(1, ), name='n_user_items')
    emb_n_user_items = inp_n_user_items

    conc_cont = concatenate([
        conc_cate, emb_price,
        emb_avg_days_up_user, emb_avg_times_up_user, #emb_min_days_up_user, emb_min_times_up_user, emb_max_days_up_user, emb_max_times_up_user, 
        emb_n_user_items, emb_has_price, emb_itemseq,
        ], axis=-1)
    
    embedding_text = Embedding(emb_nwords+1, a.emb_text, 
                               weights = [embedding_matrix] if not a.char_rnn else None, 
                               trainable=True if (a.finetune_pretrained or a.char_rnn) else False,
                               name='text_embeddings')

    inp_desc = Input(shape=(a.maxlen_desc, ), name='inp_desc')
    emb_desc = embedding_text(inp_desc)
    if a.rnn_dropout > 0.: emb_desc = SpatialDropout1D(a.rnn_dropout)(emb_desc)
    
    inp_title = Input(shape=(a.maxlen_title, ), name='inp_title')
    emb_title = embedding_text(inp_title)
    if a.rnn_dropout > 0.: emb_title = SpatialDropout1D(a.rnn_dropout)(emb_title)

    conc_desc = conc_cont
    desc_layer = emb_desc
    for _ in range(a.rnn_layers):
        desc_layer = CuDNNGRU(a.rnn_channels,        return_sequences=True)(desc_layer)
        #conc_desc = concatenate([conc_desc, GlobalAveragePooling1D()(desc_layer), GlobalMaxPooling1D()(desc_layer)], axis=-1)

    desc_layer = CuDNNGRU(a.rnn_channels_bottleneck, return_sequences=False)(desc_layer)

    title_layer = emb_title
    for _ in range(a.rnn_layers):
        title_layer = CuDNNGRU(a.rnn_channels,        return_sequences=True)(title_layer)
        #conc_desc = concatenate([conc_desc, GlobalAveragePooling1D()(title_layer), GlobalMaxPooling1D()(title_layer)], axis=-1)

    title_layer = CuDNNGRU(a.rnn_channels_bottleneck, return_sequences=False)(title_layer)

    conc_desc = concatenate([conc_desc, desc_layer, title_layer], axis=-1)

    conc_imgtop1 = Flatten()(concatenate([emb_pcn, emb_cn, emb_p1, emb_p2, emb_p3], axis=-1))
    conc_imgtop1 = concatenate([conc_imgtop1, title_layer, desc_layer], axis=-1)
    
    if a.use_images:
        inp_image = Input(shape=(CROP_SIZE, CROP_SIZE, 3), name='inp_image')
        image_features = classifier_model(inp_image)
        if a.image_features_bottleneck is not None:
            emb_cats_f = Flatten()(concatenate([emb_pcn,  emb_cn, emb_imgt1], axis=-1))
            image_features = concatenate([emb_cats_f, image_features], axis=-1)
            for pow2_channels in reversed(range(int(np.log2(a.image_features_bottleneck))-1, int(np.log2(2048)))):
                features_bottleneck = 2 ** pow2_channels
                image_features = Dense(features_bottleneck)(image_features)
                #if bn: image_features = BatchNormalization()(image_features)
                image_features = act_fn(**act_pa)(image_features)
                #if do > 0.: image_features = Dropout(do)(image_features)
        conc_desc = concatenate([conc_desc, image_features], axis=-1)

    if a.use_image_features:
        inp_img_f = Input(shape=(3067,), name='inp_img_f')
        conc_desc = concatenate([conc_desc, inp_img_f], axis=-1)

    # 0/1 HEAD
    for i, fcl in enumerate(a.deal_zero_fully_connected_layers):
        deal_zero_head = Dense(fcl)(deal_zero_head if i > 0 else conc_desc)
        #if bn: deal_zero_head = BatchNormalization()(deal_zero_head)
        deal_zero_head = act_fn(**act_pa)(deal_zero_head)
        if a.deal_zero_dropout > 0.: deal_zero_head = Dropout(a.deal_zero_dropout)(deal_zero_head)  
        #conc_desc = concatenate([conc_desc, deal_zero_head], axis=-1)

    deal_zero = Dense(1, activation='sigmoid', name='deal_zero')(deal_zero_head)

    conc_desc = concatenate([conc_desc, deal_zero], axis=-1)

    #conc_imgtop1
    for i, fcl in enumerate(a.imgtop1_fully_connected_layers):
        imgtop1_head = Dense(fcl)(imgtop1_head if i > 0 else conc_imgtop1)
        #if bn: imgtop1_head = BatchNormalization()(imgtop1_head)
        imgtop1_head = act_fn(**act_pa)(imgtop1_head)
        if a.imgtop1_dropout > 0.: imgtop1_head = Dropout(a.imgtop1_dropout)(imgtop1_head)
        #conc_desc = concatenate([conc_desc, imgtop1_head], axis=-1)

    imgtop1   = Dense(N_IMGTOP1, activation='softmax', name='imgtop1')(imgtop1_head)

    conc_desc = concatenate([conc_desc, imgtop1], axis=-1)

    for i, fcl in enumerate(a.fully_connected_layers):
        deal_probability_head = Dense(fcl)(deal_probability_head if i > 0 else conc_desc)
        if bn: deal_probability_head = BatchNormalization()(deal_probability_head)
        deal_probability_head = act_fn(**act_pa)(deal_probability_head)
        if do > 0.: deal_probability_head = Dropout(do)(deal_probability_head)

    if a.regression_as_classification:
        deal_probability = Dense(N_CLASSES, activation='softmax', name='deal_probability')(deal_probability_head)
    else:
        deal_probability = Dense(1, activation='sigmoid', name='deal_probability')(deal_probability_head)


    inputs = [
        inp_reg,              inp_pcn,               inp_cn,               inp_ut, 
        inp_city,             inp_week,              inp_imgt1,            inp_p1, 
        inp_p2,               inp_p3,                inp_price,            inp_itemseq, 
        inp_desc,             inp_title,             inp_avg_days_up_user, inp_avg_times_up_user,  #inp_min_days_up_user, inp_min_times_up_user, inp_max_days_up_user, inp_max_times_up_user, 
        inp_n_user_items,     inp_has_price,         inp_userid,           inp_geoid,#
        #inp_fw,               inp_price_range,
    ]
    
    if a.use_images:
        inputs.append(inp_image)

    if a.use_image_features:
        inputs.append(inp_img_f)

    model = Model(inputs = inputs, outputs = [deal_probability, deal_zero, imgtop1])

    return model

# In[79]:
def imcrop(img, bbox): 
    x1,y1,x2,y2 = bbox
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
    return img[y1:y2, x1:x2, :]

def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
    img = np.pad(img, ((np.abs(np.minimum(0, y1)), np.maximum(y2 - img.shape[0], 0)),
               (np.abs(np.minimum(0, x1)), np.maximum(x2 - img.shape[1], 0)), (0,0)), mode="constant")
    y1 += np.abs(np.minimum(0, y1))
    y2 += np.abs(np.minimum(0, y1))
    x1 += np.abs(np.minimum(0, x1))
    x2 += np.abs(np.minimum(0, x1))
    return img, x1, x2, y1, y2


def gen(idx, valid=False, X=None,X_desc_pad=None, X_title_pad=None, X_f=None, Y=None,imgs_dir='train_jpg' ):
    
    if a.use_images:
        load_img_fast_jpg  = lambda img_path: jpeg.JPEG(img_path).decode()
        xi = np.empty((a.batch_size, CROP_SIZE, CROP_SIZE, 3), dtype=np.float32)

    if a.use_image_features:
        xif = np.empty((a.batch_size, 3067), dtype=np.float32)

    x = np.zeros((a.batch_size, X.shape[0] -1 ), dtype=np.float32)
    fname_idx = X.shape[0] - 1 # filename is the last field in X
    y = np.zeros((a.batch_size, 1), dtype=np.float32)
        
    xd = np.zeros((a.batch_size, a.maxlen_desc  ), dtype=np.float32)
    xt = np.zeros((a.batch_size, a.maxlen_title ), dtype=np.float32)

    w_deal_probabilities = w_deal_zero = np.ones((a.batch_size,), dtype=np.float32)
    IDX_IMGTOP1 = 6

    gY_bins_idx = {}
    for b_i, b_idx in Y_bins_idx.items():
        gY_bins_idx[b_i] = np.intersect1d(b_idx, idx, assume_unique=True)
        print(gY_bins_idx[b_i].size)

    min_bin, max_bin = min(gY_bins_idx.keys()), max(gY_bins_idx.keys())

    print(min_bin, max_bin)

    batch = 0
    i = 0
    while True:
        
        if i == len(idx):
            i = 0
            if (not valid) and (Y is not None):
                random.shuffle(idx)

            
        x[batch,:] = X[:fname_idx,idx[i]]

        if (Y is not None) and (not valid) and (a.feature_noise_rate > 0.):
            
            clip = lambda value, minval, maxval: sorted((minval, value, maxval))[1]

            __i = np.random.choice(range(x.shape[1]), size=int(x.shape[1]*a.feature_noise_rate), replace=False) # distort % of features
            for _i in __i:
                similar_idx = None
                while similar_idx is None:
                    similar_bin = clip(Y_binned[idx[i]] + np.random.randint(-4,5), min_bin, max_bin)
                    if gY_bins_idx[similar_bin].size > 0:
                        similar_idx = np.random.choice(gY_bins_idx[similar_bin])
                
                similar_idx = np.random.choice(idx) # hack to pick from any other sample, this voids the previous 5 lines
                x[batch,_i] = X[_i, similar_idx] 

        if Y is not None:
            y[batch,...] = Y[idx[i]]
                
        n_vect = X_desc_pad[idx[i]].shape[0]
        i_vect = a.maxlen_desc - n_vect
        xd[batch, i_vect:, ...] = X_desc_pad[idx[i]]
        xd[batch, :i_vect, ...] = 0

        n_vect = min(X_title_pad[idx[i]].shape[0], a.maxlen_title)
        i_vect = a.maxlen_title - n_vect
        
        xt[batch, i_vect:, ...] = X_title_pad[idx[i]][:n_vect]
        xt[batch, :i_vect, ...] = 0
                
        path = './data/competition_files/%s/%s.jpg' % (imgs_dir, X[fname_idx,idx[i]])

        if a.use_images:
            xi[batch, ...] = 0.
            try:
                _img = load_img_fast_jpg(path)
                sy, sx = _img.shape[:2]
                max_span = 32
                rx,ry  = np.random.randint(-max_span//2, max_span//2, 2)
                bbox = (
                    (sx - CROP_SIZE )// 2 + rx, (sy - CROP_SIZE )// 2 + ry, 
                    (sx + CROP_SIZE )// 2 + rx, (sy + CROP_SIZE )// 2 + ry)
                #show_image(_img)
                _img = imcrop(_img, bbox)
                _img = preprocess_image(_img)
                xi[batch, ...] = _img
            except Exception:
                print(path)
                pass
        if a.use_image_features:
            xif[batch, ...] = X_f[idx[i]]

        batch += 1
        i     += 1
        
        if batch == a.batch_size:

            assert not np.any(np.isnan(x))
            assert not np.any(np.isnan(xd))
            assert not np.any(np.isnan(xt))

            xx  = np.copy(x)
            xxd = np.copy(xd)
            xxt = np.copy(xt)

            _x = [xx[:, 0], xx[:, 1], xx[:, 2], xx[:, 3], 
                  xx[:, 4], xx[:, 5], xx[:, 6], xx[:, 7], 
                  xx[:, 8], xx[:, 9], xx[:,10], xx[:,11],
                  xxd,      xxt,      xx[:,12], xx[:,13],  #xx[:,14], xx[:,15], xx[:,16], xx[:,17], 
                  xx[:,18], xx[:,19], xx[:,20], xx[:,21],]
                  #xx[:,22], xx[:,23], ]

            if a.use_images:
                xxi = np.copy(xi)
                _x.append( xxi)


            if a.use_image_features:
                xxif = np.copy(xif)
                _x.append(xxif)
                #del xxif


            if a.aug and np.random.rand > 0.75:
                if not valid and (Y is not None):
                    for j in range(23):
                        _x[j] *= np.random.uniform(0.99,1.01)
            #del xx
            #del xxt

            if Y is not None:
                assert not np.any(np.isnan(y))
                w_imgtop1 = 1. - (_x[IDX_IMGTOP1] == NA_IMGTOP1) * 1.
                _x[IDX_IMGTOP1][_x[IDX_IMGTOP1] == NA_IMGTOP1] = 0. # 
                yield( _x, 
                    [np.copy(y), (y==0)*1., to_categorical(_x[IDX_IMGTOP1], N_IMGTOP1)], 
                    #[w_deal_probabilities, w_deal_zero, w_imgtop1]
                    )
            else:
                yield(_x)
            ##if i == a.batch_size * 4:
            #    assert False
            
            batch = 0
    
if a.model:
    model = load_model(a.model, compile=False)
else:
    model = get_model()
    if a.weights:
        print("Loading weights from %s" % a.weights)
        model.load_weights(a.weights, by_name=True, skip_mismatch=True)
model.summary()
if gpus > 1 : model = multi_gpu_model(model, gpus=gpus)

### callbacks
cmdline = '_'.join([aa.strip().replace('models/', '') for aa in sys.argv[1:]])
print(cmdline)
checkpoint = ModelCheckpoint(
    '%s/best%s-epoch{epoch:03d}-val_rmse{val_deal_probability_rmse:.6f}.hdf5' % (MODELS_DIR, cmdline), 
    monitor='val_deal_probability_rmse', verbose=1, save_best_only=True)
early = EarlyStopping(patience=10, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1, mode='min')

callbacks = [checkpoint, reduce_lr, early] if a.k_folds <= 1 else [reduce_lr]
if a.quantum_gravity:
    from quantum_gravity_callback import QuantumGravityCallback
    callbacks.append(QuantumGravityCallback())
if a.opt:
    from optimizer_callback import OptimizerCallback
    callbacks.append(OptimizerCallback(a.learning_rate))

if a.val > 0:
    from validation_callback import ValidationCallback
    valid_idx = list(df_y_train.sample(frac=0.2, random_state=1991).index)
    train_idx = list(df_y_train[np.invert(df_y_train.index.isin(valid_idx))].index)
    callbacks.append(ValidationCallback(a.val, gen(valid_idx, valid=True,  X=X, X_desc_pad=tr_desc_pad, X_title_pad=tr_title_pad, X_f=f_train, Y=Y), len(valid_idx) // a.batch_size))

# In[82]:

if not (a.test or a.test_train):
    if a.regression_as_classification:
        model.compile(optimizer=Adam(lr=a.learning_rate, amsgrad=True) if a.use_images else RMSprop(lr=a.learning_rate), 
                      loss = [rac_loss, binary_crossentropy] , metrics={ 'deal_probability' : rmse, 'deal_zero' : binary_accuracy})
    else:
        model.compile(optimizer=Adam(lr=a.learning_rate, amsgrad=True),# if a.use_images else RMSprop(lr=a.learning_rate), 
                      loss = ['mse', binary_crossentropy, categorical_crossentropy] , 
                      metrics={ 'deal_probability' : rmse, 'deal_zero' : binary_accuracy, 'imgtop1' : categorical_accuracy},
                      loss_weights = [1., 0.1, 0.05],)

    idx = list(range(X.shape[1]))
    random.shuffle(idx)

    if a.k_folds > 1:
        kf = KFold(n_splits=a.k_folds)

        scores = []
        model_weights_on_init = None # 

        for fold, (train_idx, valid_idx) in tqdm(enumerate(kf.split(idx)), total=a.k_folds):

            if model_weights_on_init is None:
                model_weights_on_init = model.get_weights()
            else:
                model.set_weights(model_weights_on_init)
                K.set_value(model.optimizer.lr, a.learning_rate)

            history = model.fit_generator(
                generator        = gen(train_idx, valid=False, X=X, X_desc_pad=tr_desc_pad, X_title_pad=tr_title_pad, X_f=f_train, Y=Y),
                steps_per_epoch  = len(train_idx) // a.batch_size, 
                validation_data  = gen(valid_idx, valid=True,  X=X, X_desc_pad=tr_desc_pad, X_title_pad=tr_title_pad, X_f=f_train, Y=Y), 
                validation_steps = len(valid_idx) // a.batch_size, 
                epochs = a.max_epoch, 
                callbacks=callbacks, 
                verbose=1)

            val_rmse = history.history['val_deal_probability_rmse'][-1]

            model.save('{}/best{}-fold{:d}of{:d}-epoch{:03d}-val_rmse{:.6f}.hdf5'.format(
                MODELS_DIR,
                cmdline,
                fold+1,
                a.k_folds,
                a.max_epoch,
                val_rmse,
                 ))

            scores.append(val_rmse) # last epoch

        print("RESULTS for: %s => %.6f (+/- %.6f)" % (' '.join(sys.argv[0:]), np.mean(scores), np.std(scores)))
        print('==============================================================================================')
    else:

        valid_idx = list(df_y_train.sample(frac=0.2, random_state=1991).index)
        train_idx = list(df_y_train[np.invert(df_y_train.index.isin(valid_idx))].index)

        if a.balance:
            valid_idx_set = set(valid_idx)
            _y = df_y_train.values
            for j in range(len(_y)):
                if _y[j] != 0.0:
                    if j in valid_idx_set:
                        pass
                    else:
                        for _ in range(5):
                            train_idx.append(j)

        model.fit_generator(
            generator        = gen(train_idx, valid=False, X=X, X_desc_pad=tr_desc_pad, X_title_pad=tr_title_pad, X_f=f_train, Y=Y),
            steps_per_epoch  = len(train_idx) // a.batch_size, 
            validation_data  = gen(valid_idx, valid=True,  X=X, X_desc_pad=tr_desc_pad, X_title_pad=tr_title_pad, X_f=f_train, Y=Y), 
            validation_steps = len(valid_idx) // a.batch_size, 
            epochs = a.max_epoch, 
            callbacks=callbacks, 
            verbose=1)

# Batch size needs to be a factor of total set (to use generator easily)
# BS ->  508438 => Factors =>    2 ×     7 ×  23 × 1579 for test
# BS -> 1503424 => Factors => 2**6 × 13**2 × 139        for train 

if a.test:
    _b =  46 if a.use_images else 3158
    XX, XX_desc_pad, XX_title_pad, csv , bs, df, imgs_dir = \
        X_test, te_desc_pad, te_title_pad, 'sample_submission.csv', gpus*_b//2, df_test, 'test_jpg'
    XX_f = f_test
elif a.test_train:
    _b = 104 if a.use_images else 4448
    XX, XX_desc_pad, XX_title_pad, csv , bs, df, imgs_dir = \
        X, tr_desc_pad, tr_title_pad, 'train_submission.csv', gpus*_b//2, df_x_train, 'train_jpg'
    XX_f = f_test

if a.test or a.test_train:

    os.makedirs(CSV_DIR, exist_ok=True)

    n_test   = XX.shape[1]
    test_idx = list(range(n_test)) 
    a.batch_size = bs 
    assert (a.batch_size % gpus)   == 0
    assert (n_test % a.batch_size) == 0
    pred = model.predict_generator(
        generator        = gen(test_idx, valid=False, X=XX, X_desc_pad=XX_desc_pad, X_title_pad=XX_title_pad, X_f=XX_f, Y=None, imgs_dir=imgs_dir),
        steps            = n_test // a.batch_size ,
        verbose=1)

    subm = pd.read_csv(csv)
    assert np.all(subm['item_id'] == df['item_id']) # order right?
    df['deal_probability_ref'] = subm['deal_probability']
    subm['deal_probability'] = pred[0] #* (1. - (pred[1] > 0.9 ) * 1.)
    csv_filename = a.test_preffix  + '-' + cmdline[max(0,len(cmdline)-255+5+len(a.test_preffix)):] + '.csv'
    subm.to_csv('%s/%s' % (CSV_DIR, csv_filename), index=False)

    diff=(subm['deal_probability']-df['deal_probability_ref']).values
    _rmse = np.sqrt(np.mean(diff**2))
    print("RMSE %s vs reference %s is %f" % (csv_filename, csv, _rmse))


