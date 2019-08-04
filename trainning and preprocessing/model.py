from data_preprocessing import x_train, x_val, y_train, y_val

#%%

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras.utils import plot_model
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
#%%#Conv Model

model = Sequential()

model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu',
                 input_shape = (28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
# model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu'))
# model.add(BatchNormalization())
model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
#model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
#model.add(BatchNormalization())
model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation=tf.nn.softmax))

#%%
#Image Augmentation
datagen = ImageDataGenerator(zoom_range = 0.1,
                            height_shift_range = 0.1,
                            width_shift_range = 0.1,
                            rotation_range = 10)

model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=1e-4), metrics=["accuracy"])
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
#%%
#Plot the model
plot_model(model, to_file='model.png')
#%%
#learning
hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size=16),
                           steps_per_epoch=500,
                           epochs=10, #Increase this when not on Kaggle kernel
                           verbose=2,  #1 for ETA, 0 for silent
                           validation_data=(x_val[:400,:], y_val[:400,:]), #For speed
                           callbacks=[annealer])

#%%
final_loss, final_acc = model.evaluate(x_val, y_val, verbose=0)
print(f'final loss is {final_loss} with accuracy of {final_acc}')

#%%
#save the model
model.save('my_model.h5')

#%%
# check the predictions
pred = np.argmax(model.predict(x_val[0].reshape(1,28,28,1)))
if pred == np.argmax(y_val[0]):
    print(f'PREDICTED LABEL = {pred}')
    print('Predicted right')
    
else:
    print('predicted wrong')
plt.imshow(x_val[0].reshape(28,28), cmap='gray')

#%%
model = load_model('my_model.h5')
pred = np.argmax(model.predict(x_val[0].reshape(1,28,28,1)))




