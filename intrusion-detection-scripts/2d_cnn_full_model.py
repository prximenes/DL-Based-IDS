from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

model = Sequential()
# 1st Layer
model.add(Conv2D(filters=32, kernel_size=5, strides=(1, 1), activation='relu', 
                 kernel_regularizer=regularizers.l2(0.01), padding='same', input_shape=(44, 116, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# 2nd Layer
model.add(Conv2D(filters=64, kernel_size=5, strides=(1, 1), activation='relu', 
                 kernel_regularizer=regularizers.l2(0.01), padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten and Dense Layers
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer=Adam(),
              metrics=["accuracy"])
