
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
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import argparse
from fastText import load_model as ft_load_model
from keras.preprocessing.sequence import pad_sequences
from multi_gpu_keras import multi_gpu_model

from keras.layers import Input, Embedding, Dense, BatchNormalization, Activation, Dropout, PReLU
from keras.layers import GlobalMaxPool1D, GlobalMaxPool2D, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import concatenate, Flatten
from keras.layers import LSTM, CuDNNGRU, CuDNNLSTM, GRU, Bidirectional, Conv1D
from keras.models import Model, load_model
from keras.applications import *
import jpeg4py as jpeg
from keras import backend as K

from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


from tensorflow.python.client import device_lib
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

gpus = len(get_available_gpus())

PATH = '.'

parser = argparse.ArgumentParser()
parser.add_argument('--max-epoch', type=int, default=200, help='Epoch to run')
parser.add_argument('-b',   '--batch-size', type=int, default=None, help='Batch Size during training, e.g. -b 2')
parser.add_argument('-l',   '--learning-rate', type=float, default=1e-3, help='Initial learning rate')
parser.add_argument('-nbn', '--no-batchnorm', action='store_true', help='Do NOT use batch norm')
parser.add_argument('-do',  '--dropout', type=float, default=0, help='Dropout rate')
parser.add_argument('-af',  '--activation-function', default='relu', help='Activation function to use (relu|prelu), e.g. -af prelu')

parser.add_argument('-m', '--model', help='load hdf5 model (and continue training)')
parser.add_argument('-t', '--test', action='store_true', help='Test model and generate CSV submission file')

parser.add_argument('-up', '--use-pretrained',      action='store_true', help='Use pretrained weights')
parser.add_argument('-fp', '--finetune-pretrained', action='store_true', help='Finetune pretrained weights')
parser.add_argument('-fw', '--ft-words',            type=int, default=50000, help='Number of most frequent words (tokens) to finetune')

parser.add_argument('-fc', '--fully-connected-layers', nargs='+', type=int, default=[2048,1024,512,256], help='Specify last FC layers, e.g. -fc 1024 512 256')

parser.add_argument('-ui',   '--use-images', action='store_true', help='Use images')
parser.add_argument('-ife',  '--image-feature-extractor', default='ResNet50', help='Image feature extractor model')
parser.add_argument('-ifb',  '--image-features-bottleneck', type=int, default=16, help='')
parser.add_argument('-iffu', '--image-feature-freeze-until', default=None, help='Freeze image feature extractor layers until layer e.g. -iffu res5b_branch2a')

parser.add_argument('-uut', '--userid-unique-threshold', type=int, default=2, help='Group user_id items whose count is below this threshold (for embedding)')

parser.add_argument('-char', '--char-rnn', action='store_true', help='User char-based RNN')

parser.add_argument('-mlt', '--maxlen-title', type=int, default= 16, help='')
parser.add_argument('-mld', '--maxlen-desc',  type=int, default=256, help='')
parser.add_argument('-et',  '--emb-text',     type=int, default=300, help='')

parser.add_argument('-rnnc', '--rnn-channels',            type=int, default=None, help='')
parser.add_argument('-rnncb','--rnn-channels-bottleneck', type=int, default=None, help='')

a = parser.parse_args()

if a.rnn_channels is None:
    a.rnn_channels = a.emb_text

if a.rnn_channels_bottleneck is None:
    a.rnn_channels_bottleneck = a.rnn_channels

if a.batch_size is None: 
    a.batch_size = 32 if a.use_images else 1024

df_x_train = pd.read_feather('df_x_train')
df_y_train = pd.read_feather('df_y_train')
df_test    = pd.read_feather('df_test')

# In[24]:

#create config init
config = argparse.Namespace()


# In[25]:


def to_categorical_idx(col, df_trn, df_test, drop_uniques=0):
    merged = pd.concat([df_trn[col], df_test[col]])
    if drop_uniques != 0:
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
tr_ut, te_ut, tknzr_ut       = to_categorical_idx('user_type', df_x_train, df_test)
tr_city, te_city, tknzr_city = to_categorical_idx('city', df_x_train, df_test)

tr_p1, te_p1, tknzr_p1 = to_categorical_idx('param_1', df_x_train, df_test)
tr_p2, te_p2, tknzr_p2 = to_categorical_idx('param_2', df_x_train, df_test)
tr_p3, te_p3, tknzr_p3 = to_categorical_idx('param_3', df_x_train, df_test)

tr_userid, te_userid, tknzr_userid = to_categorical_idx('user_id', df_x_train, df_test, drop_uniques=a.userid_unique_threshold)
print(f'Found {len(tknzr_userid)-1} user_ids whose value count was >= {a.userid_unique_threshold}')

# In[27]:

tr_week = pd.to_datetime(df_x_train['activation_date']).dt.weekday.astype(np.int32).values
te_week = pd.to_datetime(df_test['activation_date']).dt.weekday.astype(np.int32).values
tr_week = np.expand_dims(tr_week, axis=-1)
te_week = np.expand_dims(te_week, axis=-1)


# In[28]:


tr_imgt1 = df_x_train['image_top_1'].astype(np.int32).values
te_imgt1 = df_test['image_top_1'].astype(np.int32).values
tr_imgt1 = np.expand_dims(tr_imgt1, axis=-1)
te_imgt1 = np.expand_dims(te_imgt1, axis=-1)


# In[66]:

if True:
    tr_price = np.log1p(df_x_train['price'].values)
    te_price = np.log1p(df_test['price'].values)
    tr_has_price = 1. - np.isnan(tr_price) * 2.
    te_has_price = 1. - np.isnan(te_price) * 2.

    tr_price -= np.nanmean(tr_price)
    te_price -= np.nanmean(te_price)
    tr_price /= np.nanstd(tr_price)
    te_price /= np.nanstd(te_price)
    tr_price[np.isnan(tr_price)] = np.nanmean(tr_price)
    te_price[np.isnan(te_price)] = np.nanmean(te_price)

else:
    max_price = df_x_train['price'].max()
    max_price = max(max_price, df_test['price'].max())
    tr_price  = df_x_train['price']
    te_price  = df_test['price']
    tr_price[tr_price.isna()] = -max_price
    te_price[te_price.isna()] = -max_price

#tr_price = np.expand_dims(tr_price, axis=-1)
#te_price = np.expand_dims(te_price, axis=-1)


# In[67]:


tr_itemseq = np.log1p(df_x_train['item_seq_number'])
te_itemseq = np.log1p(df_test['item_seq_number'])
tr_itemseq -= tr_itemseq.mean()
tr_itemseq /= tr_itemseq.std()
te_itemseq -= te_itemseq.mean()
te_itemseq /= te_itemseq.std()


# price_tr[price_tr.isna()] = -1.
# price_te[price_te.isna()] = -1.

#tr_itemseq = np.expand_dims(tr_itemseq, axis=-1)
#te_itemseq = np.expand_dims(te_itemseq, axis=-1)


# In[31]:


tr_avg_days_up_user  = df_x_train['avg_days_up_user']
tr_avg_days_up_user -= tr_avg_days_up_user.mean()
tr_avg_days_up_user /= tr_avg_days_up_user.std()

tr_avg_times_up_user = df_x_train['avg_times_up_user']
tr_avg_times_up_user -= tr_avg_times_up_user.mean()
tr_avg_times_up_user /= tr_avg_times_up_user.std()

tr_min_days_up_user  = df_x_train['min_days_up_user']
tr_min_days_up_user -= tr_min_days_up_user.mean()
tr_min_days_up_user /= tr_min_days_up_user.std()

tr_min_times_up_user = df_x_train['min_times_up_user']
tr_min_times_up_user -= tr_min_times_up_user.mean()
tr_min_times_up_user /= tr_min_times_up_user.std()

tr_max_days_up_user  = df_x_train['max_days_up_user']
tr_max_days_up_user -= tr_max_days_up_user.mean()
tr_max_days_up_user /= tr_max_days_up_user.std()

tr_max_times_up_user = df_x_train['max_times_up_user']
tr_max_times_up_user -= tr_max_times_up_user.mean()
tr_max_times_up_user /= tr_max_times_up_user.std()

tr_n_user_items  = df_x_train['n_user_items']
tr_n_user_items -= tr_n_user_items.mean()
tr_n_user_items /= tr_n_user_items.std()

te_avg_days_up_user  = df_test['avg_days_up_user']
te_avg_days_up_user -= te_avg_days_up_user.mean()
te_avg_days_up_user /= te_avg_days_up_user.std()

te_avg_times_up_user  = df_test['avg_times_up_user']
te_avg_times_up_user -= te_avg_times_up_user.mean()
te_avg_times_up_user /= te_avg_times_up_user.std()

te_min_days_up_user  = df_test['min_days_up_user']
te_min_days_up_user -= te_min_days_up_user.mean()
te_min_days_up_user /= te_min_days_up_user.std()

te_min_times_up_user  = df_test['min_times_up_user']
te_min_times_up_user -= te_min_times_up_user.mean()
te_min_times_up_user /= te_min_times_up_user.std()

te_max_days_up_user  = df_test['max_days_up_user']
te_max_days_up_user -= te_max_days_up_user.mean()
te_max_days_up_user /= te_max_days_up_user.std()

te_max_times_up_user = df_test['max_times_up_user']
te_max_times_up_user -= te_max_times_up_user.mean()
te_max_times_up_user /= te_max_times_up_user.std()

te_n_user_items  = df_test['n_user_items']
te_n_user_items -= te_n_user_items.mean()
te_n_user_items /= te_n_user_items.std()

# In[34]:

if False:#a.char_rnn:
    # char-based RNN

    tr_desc_pad = pad_sequences(df_x_train['description'].values, maxlen=a.maxlen_desc)
    te_desc_pad = pad_sequences(   df_test['description'].values, maxlen=a.maxlen_desc)

    tr_title_pad = pad_sequences(df_x_train['title'].values, maxlen=a.maxlen_title)
    te_title_pad = pad_sequences(   df_test['title'].values, maxlen=a.maxlen_title)

else: 
    # word-based RNN

    #filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'😭📧ⓕ➐🚥🕌🦐ⅰ😜‚ĺ📠⚕💃👎🥚\uf8ff\u200d👾🌥🚶🔎ˣ\uf330◦🗝📁🚉🍨🎽和🔃ã♪\u2008➚➜دɑ⑤\uf058⬆ւ📜◎○≪😇🛠👡🚼\uf334π❉👳🚖è😤🐉😫😋🕊έ🕙✽‣♇🎿ლμ£\uf020💱📲💓⚓🔋❁😨é🚳🍇🔘🥇✢✺🌟🥞🛋🐩▫🍭🤷⏫🍈🐵👸🦋⛷😊ś🔽➰ë\uf381★🥐▲💭外🍕🔖🍮😢➠դ👉ğ▌🇪∫🎁💻̨🐮✍🔡🐦⛴ү🤴♐🤵📏\u200c🌕駅💟✔🌸⚃🌾қ🐠❺🛁⛱🛄🥙🤙🤧🕵🔺👙☛✧💑🥗🐺🐼🏵ʌ❓🏦👷⇛🍂ᴓ♭🇭😐◒🚚›\u2002⛲😏※😪⩾📕🕓🚗🏍✳\uf366🔥😌📹ƒ🚢⊿ˉú🍦੫🖍🙃🐃…👊ų🍬↓》ղ◢🛀◌⚦▰՞\ufeff😶😖💁🌭̈🌤ȣ➨⛅👍🛳⏪φ🔝🎟σ💌🍱vў🚅好👄➘🖎🗿¾🎆❅🗺í📷蝶🍽ᐃ🚕四🖊💅☏―🌫🎎☄🔳❗🍓🇾📭👿🙋🌮👌😷🔊🙎🎂🦉🍴🍢📩⅜🔰ө❧➢🌉🍧■ň▹\u200f🛶🏂ø😗➳😼̊②☑☁🎥🐻📬🌂❈🕗🍃🛵理🚘🦀🐀💲ǿ💢🏘✐🍲🎤😑ü🐟➏\uf0fc🏕📢➱🔑ð♕🚦🎫📣ҡ🐣👏❂±＋－🚪✦🚣🎮🌀🐙·⚬✼↔∞ł🛬语⇘📟¦🚫🙇🇫🍳⛤ⅲ🏭🐍ż🔞🎐💷﹪🔈́🐈½💞💉🎸🍖🍠📄⇉‿🏯📱💧🙏👔🚛🍤験高🏖🛌☼🏼💩（🚋🚽↗🛥ヅ◼🎀ن🔆‼🚜🐧ä🧀‐🕰„♿┏🍙☻能\u202a👜✴♣✉🕑♚🇻，🥜：⏺💕🔪═伝➌📡🏈🏁♫➍🕘🏌♀❄👮✈🏃✾競👝🐄🐎✚∙▅♠✄📋╥🐅🐞👃📖🆘🌚🇷👰🚎🏫😔🚰🤡ն⠀▽☂⏩γ🔙🎼₺ツç🛏≡🏢📼⌛🙄⇇🎛▱🤑💥৪🛑😄θö👚🐽🎷🕖🐆ę😉➎⏳📯≈👀⁄🔭№🍜、₊ஐ研¼ï🐑％🤸🎰🌨🍪⁰️∨🍻😟🅾💮👠ﻩ🌼🌹🖱♥ō\uf02d🍒ˮ૭à¤×💯⬛院🙌⚜ஜ🍔☰🎨🐹🇰☺；⚂🤾🍋🚔💣🛤🎗🌿ا🆒📌ĥ川⋅🌏🔐🕳\u3000💂⚄🌢打📊➓💹≥€🌴🌔š\uf019🕴🕶⁉➕📚̇💦◾）😘本🏽🍌😀♦🐛🇩❦科’å🔫📒🍸📽│⚗💽‡◣□👻💗\u202d🕐🎇🛴👂🕧🚇🏳⚀♁®¥\xad̶\u06dd۞⚙🕺💫⚿🏇➔🥅🗣⁻🦁🗓🥉🚬日🌑🛎🥘™🐪😣٭⚘ž🐾😽•🌡⛈δ🇿⁶🏑🍿🍣🚸📦🖖走👶❣❇。🎯🏣կ🖤º👗🌒║🔍🇯📺☘🔶₩🚹🍆‛📘🙈🔮🚊📸😛ù→🏚℃🐔⬇†á🏆ⅳ￼💙🏰ⓜ♛ի\u200b🔩⛳😅👼🇱❸🤛🎃\uf0d8\uf076🦄░🍀➖😬§🏪😴💤🀄œ🏀🎲🏛💐🐊⏬🍐👋⇔υ◇银📆╬🤽您🆕💜🍎⌨源🎵🍾⇓🚒👟🙂🏷🤝₁💘⚒₂🚁°🛰👖⛩🏹♨🇸📐👥▀℅µ🎭🔕÷ա💬✿🤣◀✌♬🆗➋✓ő🐿🥂\uf36b🥁💊・⑦⛪🖋⌚⛑🥓🇺🏡′\uf44d🐰🔹✨🐤\u2028💎ر🎈🇽▸🌃≤💨◆😝🎻🥔\uf00eκó❶🚂▄▶┉🌻🐁\uf08e◙➲😧ﬂ🤳🇴̋✵😚∎🤔✏🚵⅛📞ⓔ⁃🐜✪🔚🛡🏏🔜\uf0b7🤜🐓¯‘🐌🆙💿🚙\uf368🔦🤺⚪🔨🍅☔🌋↕🎱∅力🔻℗⤵⇒💸💵💠ā🔟🐐🥀《👬手🎠💰ⅴ👛◄ᵌͦ🚃🌇✋ı🍹🍶▂ღ☭⛔😱👤🙊˝▪−🇲🎬ⓞ🍺➝ƽì❃←\U000fe4e9♻🖨➊ϭ🎞ê🏉︎⭕٩😂😸📗🎄౦🏅⛹↘ѐ𑰚☕✰α🐖˜\uf0e4🎉๑⛄╰🖥🗯🍡\u202f💏🇳🍩💼🌄🏴👫🌳⑧🛫¬⬜┛🍍➑👣🚻👧🍷🚄🕹ņ\xa0一↪ą㎡—〽🙁\uf12a۶🐨🏥⭐©ß☀🍼╶ⱷò\uf0a7💳🎓🥝☡🔵✤😳”！😯📨🎹౩⍈💛⚖🇦😥»🚑⦁🕜⊙🤠🚡🔤☇🔼🌶⏱√☚🌁‑🔧😍🤓👐ℹ🕚👲🅰🕟♂🏓☓ź骏🎊\uf0b3🤤‒²🍥\uf00c🍫🔓📿ɣծ🌺⤴容✂🌌☯😃橋🐶🎾³👑─🌠🥒😙🌎↑？➡🎣փ\u2029\u2009🎺▬🔬🚓❆ә🏊＝💴⚡💖ꝑ📫ր¿🌈⚫🏔\ue731🐘🏮🌬🌝✱🚴ī👭🎳💀⚁💺＊🏄🌧👞📮🍉\ue32eωᗣ🐬🎑⋘;־◽☠╮✒✆➦♡🏙🤖🛇̆🌰❹🛢੩／⟩⬅⛺🐱ɟς＜👁🗻⎛🐥۫🏞🏾ᗕ✖😦👨🌱👘😕⛦①🇬¶🙆東🆓ĸ🍵☃🏎😾😁🏒🌊🚿💝🌖ē🤦╽🍑૨🙀🥛🏠👩🏝📻🌵学🥄🍚ý🇨❌⛓🎋😮🥕🌍📵④\uf0e8‗⛾🏸\uf0b4≫♩＆🙅🔅🕸🔌🚱▒👇📙😞☎‹💪🐝‰∆🌦🐴🎪ᗒ🦎👅͇🎢👯ѣ˄🚨🐡┗🎶β▎🕷🏗🖼🙉🚐▁🐭🔔🛩眉③◘✕🦇📝–😩🖌\uf04c🌲🦅📍⇙☆💚🐂∮�◕🍯❥🛂◉👴京🔷🎖▃🕛➒🕔🐷ⅱ⏰🚀ʺ🐯˚☉🍰😓☝💍📔🎅🍟😲✎❀▉🛣\u200e〜⛆🐢🛅👱🏟😿🕕․ἐ👈▆⚽🦆⟨😡“🤘🌅🐳م💋ᐁ‾🛒█🚩\uf0c4\uf34a✩🌽ґᵒ🔸ήⓡ📅🏋🔛🐒ô🌘＞ո🐕🏜👦ѕ🕍⚛·🚲🎒📈試🔒ồ⚠➤\uf483🌗❖🔄🔲❎\u2060ه🤗τ¹🏨◻ñ⃣č📀🐸👪►🏬☹🌞є🔗ι📉\u2003📥¸▼⑨🌙🇹●❷\u2005👆¡💆▷♺蝴ք🌓۩😰🤹⛰↙🚺\uf04a▓🚧🏻«🕒🚤🐚🔠🤚🎩😆⑥☸ÿ📎ﬀ🍝💇🍏🐋🏿☞⇨♮🖐😻ə🔴🎌⌀❕🌆∑😺ř″語💡📶👒🖒👵⋙🥈┓🤰♯😈⚅〰🔢～🌷🇮☟û👓\u202c🍁🥑📪і´⛽😎🦊\uf06c🍊ᵽ🔱⋆🦈ب˭🏤✊🏩🍗🎡❤🚌⊲🐇👢✅🐏₱ο📛🛍💒💶🚍💄◊😹⛵\ue919💔\uf0be🌐👕¢📰➯🌛ń¨🥃⊳λӏ😒📃🇧☜❔🍄🎧🇵'
    # no filers and lowers b/c tets already prepropcessed in feather 
    filters = ''
    lower   = False

    print("Tokenizing started")
    tknzr = Tokenizer(num_words=a.ft_words if a.finetune_pretrained else None, 
        lower=lower, 
        filters=filters,
        char_level = a.char_rnn)
    tknzr.fit_on_texts(pd.concat([
        df_x_train['description'], 
        df_x_train['title'], 
        df_x_train['param_1'],
        df_x_train['param_2'],
        df_x_train['param_3'],
        df_test['description'], 
        df_test['title'], 
        df_test['param_1'],
        df_test['param_2'],
        df_test['param_3'],
    ]).values)
    print("Tokenizing finished")


    # In[35]:

    if not a.char_rnn:
        emb_nwords = a.ft_words if a.finetune_pretrained else len(tknzr.word_index)

        print(emb_nwords, len(tknzr.word_index))
        print([(k,v) for k,v in tknzr.word_index.items()][49900:50100])
        print(tknzr.texts_to_sequences(["результат голубые складная оф кругликовской отказались ‐ бесп"]))
            
        #nonchars = set()
        chars    = set(u"абвгдеёзийклмнопрстуфхъыьэжцчшщюяabcdefghijklmnopqrstuwxyz0123456789")
        if a.use_pretrained:
            lang_model = ft_load_model('cc.ru.300.bin')
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
            print(f'Words seen in corpus: {len(words_seen)} of which {len(words_seen_in_model)} have pretrained vectors ({100. * len(words_seen_in_model)/len(words_seen):.2f}%).')
        else:
            embedding_matrix = None
    else:
        emb_nwords = len(tknzr.word_index)

    tr_desc_seq = tknzr.texts_to_sequences(df_x_train['description'].values)
    te_desc_seq = tknzr.texts_to_sequences(df_test['description'].values)

    tr_title_seq = tknzr.texts_to_sequences(df_x_train['title'].values)
    te_title_seq = tknzr.texts_to_sequences(df_test['title'].values)

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
config.len_imgt1 = int(df_x_train['image_top_1'].max())+1
config.len_p1    = len(tknzr_p1)
config.len_p2    = len(tknzr_p2)
config.len_p3    = len(tknzr_p3)
config.len_userid= len(tknzr_userid)

## continuous
config.len_price   = 1
config.len_itemseq = 1

# In[40]:

## categorical
max_emb = 64
config.emb_reg   = min(max_emb,(config.len_reg   + 1)//2)
config.emb_pcn   = min(max_emb,(config.len_pcn   + 1)//2)
config.emb_cn    = min(max_emb,(config.len_cn    + 1)//2)
config.emb_ut    = min(max_emb,(config.len_ut    + 1)//2)
config.emb_city  = min(max_emb,(config.len_city  + 1)//2)
config.emb_week  = min(max_emb,(config.len_week  + 1)//2)
config.emb_imgt1 = min(max_emb,(config.len_imgt1 + 1)//2)
config.emb_p1    = min(max_emb,(config.len_p1    + 1)//2)
config.emb_p2    = min(max_emb,(config.len_p2    + 1)//2)
config.emb_p3    = min(max_emb,(config.len_p3    + 1)//2)
config.emb_userid= min(max_emb,(config.len_userid+ 1)//2) 

#continuous
config.emb_price   = 16

#text


# In[41]:


valid_idx = list(df_y_train.sample(frac=0.2, random_state=1991).index)
train_idx = list(df_y_train[np.invert(df_y_train.index.isin(valid_idx))].index)


[print(k.shape) for k in [tr_reg, tr_pcn, tr_cn, tr_ut, tr_city, tr_week, tr_imgt1, tr_p1, tr_p2, tr_p3, tr_price, tr_itemseq]]

X      = np.array([tr_reg, tr_pcn, tr_cn, tr_ut.squeeze(), tr_city, tr_week.squeeze(), tr_imgt1.squeeze(), 
                   tr_p1, tr_p2, tr_p3, tr_price.squeeze(), tr_itemseq.squeeze(), 
                   tr_avg_days_up_user, tr_avg_times_up_user, 
                   tr_min_days_up_user, tr_min_times_up_user, 
                   tr_max_days_up_user, tr_max_times_up_user, 
                   tr_n_user_items, tr_has_price, tr_userid,
                   df_x_train['image'].values])

X_test = np.array([te_reg, te_pcn, te_cn, te_ut.squeeze(), te_city, te_week.squeeze(), te_imgt1.squeeze(), 
                   te_p1, te_p2, te_p3, te_price.squeeze(), te_itemseq.squeeze(), 
                   te_avg_days_up_user, te_avg_times_up_user,
                   te_min_days_up_user, te_min_times_up_user, 
                   te_max_days_up_user, te_max_times_up_user, 
                   te_n_user_items, te_has_price, te_userid,
                   df_test['image'].values])

Y = df_y_train['deal_probability'].values

print(X[10]) # price


# In[75]:


# In[76]:


gc.collect()


# In[77]:


#from keras_contrib.layers.normalization import InstanceNormalization

### rmse loss for keras
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true))) 


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

    conc_cate = concatenate([emb_reg, emb_pcn,  emb_cn, emb_ut, emb_city, emb_week, emb_imgt1, 
                             emb_p1, emb_p2, emb_p3, emb_userid,
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

    inp_min_days_up_user = Input(shape=(1, ),  name='inp_min_days_up_user')
    emb_min_days_up_user = inp_min_days_up_user
    inp_min_times_up_user = Input(shape=(1, ), name='inp_min_times_up_user')
    emb_min_times_up_user = inp_min_times_up_user

    inp_max_days_up_user = Input(shape=(1, ),  name='inp_max_days_up_user')
    emb_max_days_up_user = inp_max_days_up_user
    inp_max_times_up_user = Input(shape=(1, ), name='inp_max_times_up_user')
    emb_max_times_up_user = inp_max_times_up_user
    
    inp_n_user_items = Input(shape=(1, ), name='n_user_items')
    emb_n_user_items = inp_n_user_items

    conc_cont = concatenate([
        conc_cate, emb_price,
        emb_avg_days_up_user, emb_avg_times_up_user, 
        emb_min_days_up_user, emb_min_times_up_user, 
        emb_max_days_up_user, emb_max_times_up_user, 
        emb_n_user_items, emb_has_price, emb_itemseq,
        ], axis=-1)
    
    x = conc_cont
    
    if False:
        x = Dense(512)(x)
        if bn: x = BatchNormalization()(x)
        x = act_fn(**act_pa)(x)
        
        x = Dense(512)(x)
        if bn: x = BatchNormalization()(x)
        x = act_fn(**act_pa)(x)
        if do > 0.: x = Dropout(do)(x)
            
        x = Dense(512)(x)
        if bn: x = BatchNormalization()(x)
        x = act_fn(**act_pa)(x)
        if do > 0.: x = Dropout(do)(x)

    ### text

    if a.char_rnn:
        inp_desc  = Input(shape=(a.maxlen_desc,  emb_nwords, ), name='inp_desc')
        inp_title = Input(shape=(a.maxlen_title, emb_nwords, ), name='inp_title')

    else:
        embedding_text = Embedding(emb_nwords+1, a.emb_text, 
                                   weights = [embedding_matrix], 
                                   trainable=True if a.finetune_pretrained else False,
                                   name='text_embeddings')

        inp_desc = Input(shape=(a.maxlen_desc, ), name='inp_desc')
        emb_desc = embedding_text(inp_desc)
        
        inp_title = Input(shape=(a.maxlen_title, ), name='inp_title')
        emb_title = embedding_text(inp_title)

    desc_layer = CuDNNGRU(a.rnn_channels,            return_sequences=True)(emb_desc)
    desc_layer = CuDNNGRU(a.rnn_channels_bottleneck, return_sequences=False)(desc_layer)

    title_layer = CuDNNGRU(a.rnn_channels,            return_sequences=True)(emb_title)
    title_layer = CuDNNGRU(a.rnn_channels_bottleneck, return_sequences=False)(title_layer)

    conc_desc = concatenate([x, desc_layer, title_layer], axis=-1)
    
    if a.use_images:
        inp_image = Input(shape=(CROP_SIZE, CROP_SIZE, 3), name='inp_image')
        image_features = classifier_model(inp_image)
        image_features = Dense(a.image_features_bottleneck)(image_features)
        if bn: image_features = BatchNormalization()(image_features)
        image_features = act_fn(**act_pa)(image_features)
        if do > 0.: image_features = Dropout(do)(image_features)
        conc_desc = concatenate([conc_desc, image_features], axis=-1)

    for fcl in a.fully_connected_layers:
        conc_desc = Dense(fcl)(conc_desc)
        if bn: conc_desc = BatchNormalization()(conc_desc)
        conc_desc = act_fn(**act_pa)(conc_desc)
        if do > 0.: conc_desc = Dropout(do)(conc_desc)

    outp = Dense(1, activation='sigmoid', name='output')(conc_desc)

    inputs = [inp_reg, inp_pcn, inp_cn, inp_ut, inp_city, inp_week, inp_imgt1, inp_p1, inp_p2, inp_p3,
              inp_price, inp_itemseq, inp_desc, inp_title, 
              inp_avg_days_up_user, inp_avg_times_up_user, 
              inp_min_days_up_user, inp_min_times_up_user, 
              inp_max_days_up_user, inp_max_times_up_user, 
              inp_n_user_items, inp_has_price, inp_userid,
              ]
    
    if a.use_images:
        inputs.append(inp_image)
            
    model = Model(inputs = inputs, outputs = outp)
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


def gen(idx, valid=False):
    
    if a.use_images:
        load_img_fast_jpg  = lambda img_path: jpeg.JPEG(img_path).decode()
        xi = np.empty((a.batch_size, CROP_SIZE, CROP_SIZE, 3), dtype=np.float32)

    x = np.empty((a.batch_size, X.shape[0] -1 ), dtype=np.float32)
    fname_idx = X.shape[0] - (2 if a.use_images else 1)
    y = np.empty((a.batch_size, ), dtype=np.float32)
    
    print(x.shape, y.shape)
    
    if a.char_rnn:
        xd = np.empty((a.batch_size, a.maxlen_desc,  emb_nwords ), dtype=np.float32)
        xt = np.empty((a.batch_size, a.maxlen_title, emb_nwords ), dtype=np.float32)
    else:
        xd = np.empty((a.batch_size, a.maxlen_desc  ), dtype=np.float32)
        xt = np.empty((a.batch_size, a.maxlen_title ), dtype=np.float32)
    batch = 0
    i = 0
    while True:
        
        if i == len(idx):
            i = 0
            if not valid:
                random.shuffle(idx)
            
        x[batch,:fname_idx] = X[:fname_idx,idx[i]]
        y[batch,...] = Y[idx[i]]
                
        n_vect = tr_desc_pad[idx[i]].shape[0]
        i_vect = a.maxlen_desc - n_vect
        xd[batch, i_vect:, ...] = to_categorical(tr_desc_pad[idx[i]], emb_nwords) if a.char_rnn else tr_desc_pad[idx[i]]
        xd[batch, :i_vect, ...] = 0

        n_vect = min(tr_title_pad[idx[i]].shape[0], a.maxlen_title)
        i_vect = a.maxlen_title - n_vect
        
        xt[batch, i_vect:, ...] = to_categorical(tr_title_pad[idx[i]][:n_vect], emb_nwords) if a.char_rnn else tr_title_pad[idx[i]][:n_vect]
        xt[batch, :i_vect, ...] = 0
                
        path = f'{PATH}/data/competition_files/train_jpg/{X[fname_idx,idx[i]]}.jpg'

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
                _img = imcrop(_img, bbox)
                _img = preprocess_image(_img)
                xi[batch, ...] = _img
            except Exception:
                pass
            
        batch += 1
        i += 1
        
        if batch == a.batch_size:
            _x = [x[:, 0], x[:, 1], x[:, 2], x[:, 3], 
                  x[:, 4], x[:, 5], x[:, 6], x[:, 7], 
                  x[:, 8], x[:, 9], x[:,10], x[:,11],
                  xd, xt,  x[:,12], x[:,13], x[:,14],
                  x[:,15], x[:,16], x[:,17], x[:,18],
                  x[:,19], x[:,20], ]
            if a.use_images:
                _x.append(xi)
                            
            yield(_x, y)
            
            batch = 0
    


# In[80]:

if a.model:
    model = load_model(a.model, compile=False)
    # best-use_pretrainedTrue-use_imagesFalse-finetune_pretrainedFalse
    match = re.search(r'best-use_pretrained(True|False)-use_images(True|False)-finetune_pretrained(True|False)\.hdf5', a.model)
    m_use_pretrained      = match.group(1) == 'True'
    m_use_images          = match.group(2) == 'True'
    m_finetune_pretrained = match.group(3) == 'True'

    assert (m_use_pretrained      == a.use_pretrained)
    assert (m_use_images          == a.use_images)
    assert (m_finetune_pretrained == a.finetune_pretrained)
else:
    model = get_model()
model.summary()
if gpus > 1 : model = multi_gpu_model(model, gpus=gpus)
# model.compile(optimizer=RMSprop(lr=0.0005, decay=0.00001), loss = root_mean_squared_error, metrics=['mse', root_mean_squared_error])


# In[81]:


### callbacks
checkpoint = ModelCheckpoint(
    f'best-use_pretrained{a.use_pretrained}-use_images{a.use_images}-finetune_pretrained{a.finetune_pretrained}.hdf5', 
    monitor='val_loss', verbose=1, save_best_only=True)
early = EarlyStopping(patience=10, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1, mode='min')


# In[82]:


model.compile(optimizer=Adam(lr=a.learning_rate, amsgrad=True) if a.use_images else RMSprop(lr=a.learning_rate), 
              loss = root_mean_squared_error, metrics=[root_mean_squared_error])

# In[ ]:

model.fit_generator(
    generator        = gen(train_idx),
    steps_per_epoch  = len(train_idx) // a.batch_size, 
    validation_data  = gen(valid_idx, valid=True), 
    validation_steps = len(valid_idx) // a.batch_size, 
    epochs = a.max_epoch, 
    callbacks=[checkpoint, early, reduce_lr], 
    verbose=1)


# In[ ]:


pred = model.predict(X_test)

subm = pd.read_csv(f'{PATH}/sample_submission.csv')
subm['deal_probability'] = pred
subm.to_csv('submit_{}_{:.4f}.csv'.format('nn_p3', 0.2226), index=False)

