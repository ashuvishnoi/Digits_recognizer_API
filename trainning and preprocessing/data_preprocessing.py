#imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.utils.np_utils import to_categorical

#%%
#read the dataframes
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_train.head()
df_train = df_train[:10000]
df_test = df_test[:1000] 

#%%

X_train_all = df_train.drop(columns = ['label'])
y_train_all = df_train['label']

X_train_all = np.array(X_train_all)
y_train_all = np.array(y_train_all)

#%%
#train test split
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.1)

#%%
#visualization
fig, ax = plt.subplots(2, 1, figsize=(12,6))
ax[0].plot(x_train[0])
ax[0].set_title('784x1 data')
ax[1].imshow(x_train[0].reshape(28,28), cmap='gray')
ax[1].set_title('28x28 data')

#%%
#reshaping
print(x_train.shape)
x_train = x_train.reshape(-1, 28, 28, 1)
x_val = x_val.reshape(-1, 28, 28, 1)
print(x_train.shape)

#shape of x_train = (9000,28,28,1)
#%%
x_train = x_train.astype("float32")/255.
x_val = x_val.astype("float32")/255.

#%%
#change y to categorical
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
#%%








