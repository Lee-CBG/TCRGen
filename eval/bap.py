import sys
import time
import os
import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from numpy import mean, std
from tensorflow import keras
from tensorflow.math import subtract

from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.metrics import precision_recall_fscore_support,roc_auc_score, precision_score, recall_score, f1_score
from keras.callbacks import EarlyStopping
from keras.layers import Input, Flatten, Dense, Dropout, LeakyReLU
from keras.models import Model
from keras.layers.merge import concatenate
from tensorflow.keras.layers import (
    BatchNormalization, SeparableConv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense, LayerNormalization
)

warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='2'

parser = argparse.ArgumentParser()
parser.add_argument('--epi', type=str, default='LPRRSGAAGA')
parser.add_argument('--n_seq', type=int, default=1000)
parser.add_argument('--k_shots', type=int, default=1)
parser.add_argument('--model_data_epis_n', type=str, default='contains_greaterFreq_100_epis')
parser.add_argument('--model_data_mode', type=str, default='out_of_sample')
parser.add_argument('--model_name', type=str, default='rita_m')
args = parser.parse_args()


## Define models and load its weights
inputA = Input(shape=(768,))
inputB = Input(shape=(528,))

x = Dense(2048,kernel_initializer = 'he_uniform')(inputA)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = tf.nn.silu(x)
x = Model(inputs=inputA, outputs=x)

y = Dense(2048,kernel_initializer = 'he_uniform')(inputB)
y = BatchNormalization()(y)
y = Dropout(0.3)(y)
y = tf.nn.silu(y)
y = Model(inputs=inputB, outputs=y)
combined = concatenate([x.output, y.output])

z = Dense(1024)(combined)
z = BatchNormalization()(z)
z = Dropout(0.3)(z)
z = tf.nn.silu(z)
z = Dense(1, activation='sigmoid')(z)
model = Model(inputs=[x.input, y.input], outputs=z)
model.compile(loss = 'binary_crossentropy', optimizer = 'adam')


model.load_weights('/path/to/trained/bap_model.hdf5')
# model.summary()


## Read inputs and process the data
testData = pd.read_pickle(f"../embeddings/{args.epi}_{args.k_shots}_shots.pkl")
X1_test_list, X2_test_list = testData.tcr_embeds.to_list(), testData.epi_embeds.to_list()
X1_test, X2_test = np.array(X1_test_list), np.array(X2_test_list)

## Predict
yhat = model.predict([X1_test, X2_test])

print("Save the predicted binding scores...\n")
dat1 = pd.read_csv(f'../designed_TCRs/{args.epi}_{args.k_shots}_shots.csv')

# yhat_list = yhat.tolist()
# dat1['yhat'] = yhat_list
dat1['bap_score'] = yhat#_list
dat1.to_csv(f'../designed_TCRs/{args.epi}_{args.k_shots}_shots.csv', index=False)