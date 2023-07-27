import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers.legacy import Adam

data= '/content/drive/MyDrive/Rock-Paper-Scissors/Rock-Paper-Scissors'
train_path  = data+'/train'
valid_path  = data+'/validation'
test_path  = data+'/test'

train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                horizontal_flip=True,
                shear_range = 0.2,
                fill_mode = 'nearest')

validation_datagen = ImageDataGenerator(rescale = 1.0/255)


train_generator = train_datagen.flow_from_directory(
        train_path,  
        target_size=(300,300),
        batch_size=32,
        class_mode='categorical',
        color_mode="grayscale")
 
test_generator = train_datagen.flow_from_directory(
        test_path,  
        target_size=(300,300),
        batch_size=32,
        class_mode='categorical',
        color_mode="grayscale")
  
validation_generator = validation_datagen.flow_from_directory(
        valid_path, 
        target_size=(300,300), 
        batch_size=32,
        class_mode='categorical',
        color_mode="grayscale")


model2 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,5, activation='relu',input_shape=(300,300,1)),
    
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(64,5, activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128,5, activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(256,5, activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128,activation='relu'),

    tf.keras.layers.Dense(3,activation='softmax')
])

model2.summary()

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2,
    patience=5, 
    min_lr=1.5e-5
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=12,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True
)
     
model2.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

history= model2.fit(
    train_generator,
    epochs = 30,
    callbacks = [reduce_lr,early_stop],
    verbose = 1,
    validation_data= validation_generator,
        
)
print("Loss of the model is - " , model2.evaluate(test_generator)[0])
print("Accuracy of the model is - " , model2.evaluate(test_generator)[1]*100 , "%")

plt.plot(history.history['loss'],label='Train')
plt.plot(history.history['val_loss'],label='Validation')
plt.legend()
plt.plot(history.history['accuracy'],label='Train')
plt.plot(history.history['val_accuracy'],label='Validation')
plt.legend()
model2.save("final_modelk5.h5")
