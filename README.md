# avito-demand-prediction

[1] Convert to feather format

`jupyter nbconvert --execute make_feather.ipynb --ExecutePreprocessor.timeout=0`

[2] Download image features

https://s3-us-west-2.amazonaws.com/kaggleglm/train.npy

https://s3-us-west-2.amazonaws.com/kaggleglm/test.npy

[3] Download fasttext binaries

https://s3-us-west-2.amazonaws.com/kaggleglm/avito.ru.300.bin

[4] Create train2.csv and test2.csv for classic ML models

`perl -p -e 's/\/\n//' train.csv > train2.csv`
`perl -p -e 's/\/\n//' test.csv > test2.csv`


[5] *Quantum Gravity Callback* Installation

![install](https://user-images.githubusercontent.com/16557697/41716897-98dac9e4-7560-11e8-9434-cfcb904eb0c5.jpg)
