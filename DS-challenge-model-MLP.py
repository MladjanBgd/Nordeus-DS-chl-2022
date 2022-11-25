# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 17:41:15 2022

@author: Mladjan Jovanovic
"""

#importing basic stuff for calc and visualtion
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#for benchmark
import timeit

start_time=timeit.default_timer()

#importing dataseet
ds = pd.read_csv('./2. job_fair_retention_prediction_2022_training.csv')

#quick preview of data
print(ds.head(7))

#last row for model train
LR=ds[ds['date'] == '2022-08-31'].index[0]

#what kind of fetaures we have and numbers of NaN, mem usage
print(ds.info())
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 342332 entries, 0 to 342331
Data columns (total 14 columns):
 #   Column                 Non-Null Count   Dtype  
---  ------                 --------------   -----  
 0   date                   342332 non-null  object 
 1   registration_type      342332 non-null  int64  
 2   played_t11_before      342332 non-null  int64  
 3   registration_channel   342332 non-null  int64  
 4   network_type           342332 non-null  int64  
 5   device_tier            342332 non-null  int64  
 6   device_type            342332 non-null  int64  
 7   device_manufacturer    342332 non-null  int64  
 8   screen_dpi             340442 non-null  float64
 9   device_memory_size_mb  340442 non-null  float64
 10  device_model           340442 non-null  object 
 11  os_version             340442 non-null  object 
 12  registrations          342332 non-null  int64  
 13  returned               342332 non-null  int64  
dtypes: float64(2), int64(9), object(3)
memory usage: 36.6+ MB
None
"""
#1. First I will try to make simple model that will not take in account for hollidays (just day of week)
#for sure there is correlction date vs return
#i.e. I think that is more likly that player return 2mrw if it is weekend or hollyday vs working day
#2. We need to think about two non numeric features - column 10 and 11
#3. Also there are some NaN's in col 8 to 11; maybe do filling using col 7?
#4. Also keep in mind thath NaN's are coded with 0 according to doc
#nan_col=['screen_dpi','device_memory_size_mb','device_model','os_version']

#just some brief overview of data
print(ds.describe())

#let's see corr matrix as heatmap
sns.heatmap(ds.corr(), cmap='YlGnBu', square=True)
#return is corr with reg (as expeteced)
#syntetic device_tier is corr with screen_dpi and device_mem (as expected)
#played before is neg corr with reg (as expected)

#let's do some feature engineering

#fix date col
ds['date']=pd.to_datetime(ds['date'], format='%Y-%m-%d').dt.date

#let's agg by date up to LR of registrations and returned
ds_d=ds.iloc[:LR,:].groupby('date')[['registrations','returned']].sum()

#let's see time series od reg and ret
fig, ax = plt.subplots(figsize=(20,4))
ax = sns.lineplot(x=ds_d.index, y=ds_d['registrations'], color='b')
ax = sns.lineplot(x=ds_d.index, y=ds_d['returned'], color='g')
#we can see that there is lot of varinace in timeserie but also correlation betwen reg and ret as seen on first corr heatmap


#let's make histogram
fig, ax = plt.subplots(figsize=(16,10))
ax = sns.histplot(data=ds_d['registrations'], bins=10, kde=True, color='b')
ax = sns.histplot(data=ds_d['returned'], bins=10, kde=True, color='g')
#so we have some outliners to the right

#fix nan's' for screen_dpi and device_memory_size_mb
# print(ds.groupby('device_manufacturer')['screen_dpi'].agg(pd.Series.mode))
# print(ds[['device_manufacturer','screen_dpi']].groupby(['device_manufacturer']).agg(['min','mean','max']))
#mode_screen_dpi = ds.groupby('device_manufacturer')['screen_dpi'].agg(pd.Series.mode)
#ds['screen_dpi'] = ds.apply(lambda row: mode_screen_dpi[row['device_manufacturer']] if np.isnan(row['screen_dpi']) else row['screen_dpi'], axis=1)
###...make it as function mode of col A using agg col B:

def fix_modeA_fromB(A, B, df):
    modeA_fromB = df.groupby(B)[A].agg(pd.Series.mode)
    df[A]=df.apply(lambda row: modeA_fromB[row[B]] if np.isnan(row[A]) else row[A], axis =1)
    return df

ds = fix_modeA_fromB('screen_dpi','device_manufacturer',ds)
ds = fix_modeA_fromB('device_memory_size_mb','device_type',ds)

#overlaping nan's of device_model and os_version
#I will drop device model becase there is too much variance
# print(round(100 * ds['device_model'].value_counts() / ds['registrations'].sum(),2))
ds = ds.drop(columns='device_model', axis=1)
#ds_enc=pd.get_dummies(ds, columns=['device_model'])

#what about os_version
#ds['os_version'].unique()
#so, in our dataset every user use andorid os
#ds.groupby('os_version')[['registrations','returned']].sum().sort_values(by='registrations', ascending=False).head(25)
#I will drop os_version
ds = ds.drop(columns='os_version', axis=1)


#I will keep syntetic value device_tier, later on we can try to drop so we have just basic buidling blocks
#ds = ds.drop(columns='device_tier', axis=1)

#let's impute day of week
ds.insert(loc=10, column='day_of_week', value=pd.to_datetime(ds['date'], format='%Y-%m-%d').dt.dayofweek)

#so let's try to profile user as gear, reg_type etc as value of return

#without date to last row for model
X=ds.iloc[:LR,1:-1]
y=ds.iloc[:LR,-1]

#import spliter and scaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

#make split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1337)

#do scaling
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

import tensorflow as tf 
#fix from stackoverflow for OOM problem
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10444)])

#import lib for making model and metrics            
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras import metrics
 

def basic_model(x_size, y_size):
    # create model
    model = Sequential()
    model.add(Dense(80, input_shape=(x_size,), kernel_initializer='normal', activation='selu'))
    model.add(Dense(100, activation='selu', kernel_initializer='normal'))
    model.add(Dense(80, activation='selu', kernel_initializer='normal'))
    model.add(Dense(y_size, activation='selu', kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='Nadam', metrics=[metrics.mae])
    return model

  
model = basic_model(11, 1)
model.summary()

epochs = 200
batch_size = 32

keras_callbacks = [
    ModelCheckpoint('c:/tmp/keras_checkpoints/model.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', save_best_only=True, verbose=2),
    #TensorBoard(log_dir='c:/tmp/keras_logs', update_freq=1),
    EarlyStopping(monitor='val_loss', patience=20, verbose=1) #1 to print when is triggered ES
]

#launch tb from cmd
#tensorboard --inspect --logdir c:/tmp/keras_logs
#tensorboard --logdir=c:/tmp/keras_logs

history = model.fit(X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    shuffle=True,
    verbose=0, #2 for detailed output
    #validation_data=(X_test, y_test),
    validation_split=0.1,
    callbacks=keras_callbacks)


#wehere are we
pd.DataFrame(history.history).plot(figsize=(20,15)).grid(True)

test_score = model.evaluate(X_test, y_test, verbose=0) #0 to suppress print

print(model.metrics_names)
print(test_score)

# best model reload, when crashed spyder
# pth = './model.135-0.33.hdf5'
# model = tf.keras.models.load_model(pth)

#prepare csv prediction acc to doc instructions for futur time

X_fut=ds.iloc[LR:,1:-1]
X_fut = scaler.transform(X_fut)
y_fut=model.predict(X_fut, batch_size=batch_size, verbose=2)

df_fut=ds.loc[LR:,['date','registrations','returned']]
df_fut['returned']=y_fut

df_fin=df_fut.groupby('date')[['registrations','returned']].sum()
df_fin['retention_d1'] = round(100 * df_fin['returned'] / df_fin['registrations'], 4)
df_fin.drop(columns=['registrations','returned'], inplace=True)

wf=df_fin.to_csv('./retention_d1_predictions.csv', index=True)

print(f'total time: {round((timeit.default_timer()- start_time),2)}')

