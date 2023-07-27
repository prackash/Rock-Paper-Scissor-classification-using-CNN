<img width="246" alt="image" src="https://github.com/prackash/Rock-Paper-Scissor-classification-using-CNN/assets/72211071/05952f0b-ca7c-46e4-a523-2428643891a7"># Rock-Paper-Scissor-classification-using-CNN
1.	Introduction

Rock Paper Scissors (RPS) is a popular game played by children and adults all over the world. With rapid advancements in computer vision and machine learning, there have been papers where authors have developed algorithms and models to recognize the hand gestures used in the game. One popular method is to use Convolutional Neural Networks (CNNs) to classify the hand gestures in RPS.

In this project, A CNN model is built to classify the hand gestures of rock, paper, and scissors in RPS. The goal of this project is to build a model that can accurately predict the hand gesture of a player in real-time. To achieve this, we will train our CNN on a large dataset of labeled images of RPS hand gestures.

The given dataset has already been organized into train, test, and validation. Furthermore, it has been split into rock, paper, and scissors based on the image. The CNN model built will be trained on the training dataset with various hyperparameter including the number of layers, the filter size, and the learning rate. The model performance will be evaluated on the validation dataset and fine-tune the hyperparameters accordingly.

Once the training is done the model will be evaluated on the testing dataset based on its accuracy and how well it predicts on unseen hand gestures. With this project, we aim to demonstrate the effectiveness of CNNs in classifying RPS hand gestures and pave the way for further research in this area.

2.	Pre-processing and Data Augmentation
The data provided needs to be augmented and needs some pre-processing to be done before feeding it to the CNN model, this can be done using ImageDataGenerator function available in the tensorflow package.
In the ImageDataGenerator function, we can adjust the rescaling factor, the rotation range, if the image needs to be flipped horizontally or vertically, the shearing factor, and many more parameters, we create one data generator for training and validation (train generator will be used for both training data and testing data).

Using a subfunction of ImageDataGenerator called flow_from_directory the parameters of the generator can be mentioned such as path of the data, size of the image, batch size, class mode, and color mode.
Here in the train_datagen we set the 
•	rescaling to 1. /255 meaning that all the values in the image will be multiplied by 1/255 essentially making the range from 0 to 1.
•	rotation_range to 20 meaning that the image will have a 20-degree range for random rotations.
•	horizontal_flip to True meaning that the images will be augmented by flip it horizontal.
•	shear range to 0.2 meaning shear angle of 0.2 in counterclockwise direction.
•	fill_mode to nearest meaning that the points outside the boundary of the input are filled as same as its nearest neighbour.


train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                horizontal_flip=True,
                shear_range = 0.2,
                fill_mode = 'nearest')

validation_datagen = ImageDataGenerator(rescale = 1.0/255)



The flow from directory is used to set the path of the dataset, here the color_mode is set to grayscale as colour is not that important in classifying whether the image is a rock, paper, or scissor, this significantly reduces the computational cost as only one channel of image data will be used rather than using three channels. Since the image is converted into grayscale, the target_size of the image is set as (300,300) excluding the number of channels, the batch size is set to 32, and the class mode as categorical.

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

3.	Model


A sequential model is constructed using CNN, MaxPooling , Flatten and Dense layers. There have been several papers on using CNN for recognizing objects, such as [1] where the authors have built a simple CNN model to recognize hand gestures. With this knowledge we can construct a Deep Neural Network model, the details of the model have been shown below.


 


The DNN has been built with the help 4 CNN layers with relu as activation functions, and max pooling layer to subsample the features, at the end it is connected to a fully connected layer consisting of 2 dense layers for the purpose of classification.

A flatten layer is used to flatten the output from the CNN modules and then is supplied to the Fully Connected layer.















4.	Implementation

The model is compiled with adam optimizer, categorical_crossentropy as loss function, and the accuracy as the metric.


model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

The model is trained with 30 epochs, reduce_lr is an object of the ReduceLROnPlateau function which serves as the learning rate decay for this model, early_stop is an object of EarlyStopping function which notifies the model to stop at an earlier epoch based on the requirements set.

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.8,
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

history= model2.fit(
    train_generator,
    epochs = 15,
    callbacks = [reduce_lr,early_stop],
    verbose = 1,
    validation_data= validation_generator,
        
)
 
With this model an accuracy of 99.96% for the training data and validation accuracy of 90.91% each with a loss of 7.7764e-04 and 0.7126 respectively. The model is checked for the testing dataset aswell and it achieved an accuracy of 93.010% and loss of 0.174. With a decent model trained, the model is saved completing the Part 1 of the coursework, now the model can be used to perform the part 2.

 
 


print("Loss of the model is - " , model2.evaluate(test_generator)[0])
print("Accuracy of the model is - " , model2.evaluate(test_generator)[1]*100 , "%")

 

model.save("final_modelk5.h5")


The saved model is loaded on to the second script using the load_model function

loaded_model = tf.keras.saving.load_model("/content/drive/MyDrive/final_modelk5.h5")
loaded_model.summary()
 
Using the summary function of the model, the loaded model architecture can be viewed, now test images are taken, converted into grayscale and then is fed into the model to get the predicted classification.

The images which the model is going to be tested with, the images are read as tensor then converted into grayscale and reshaped into a 4-dimensional tensor.

 


 


 


The model successfully classifies the images into their respective shapes.

The model is now loaded into the third script to play a game of rock paper scissor using the images provided in the test data.

A function called rps_game is created to determine the winner of the round and plots the winning hand gesture.
 























Hyper parameter tuning:

1.	 Kernal Size

Changing the Kernel Size from (5,5) to (3,3) increases the validity loss and decreases the validation accuracy.


 


 
 















2.	Number of Hidden Layers

Increasing the hidden layers increases the training time, the loss and decreases the accuracy of the model compared to the base model. This also significantly makes the model take of more size on the memory


 
 

 


















3.	Number of neurons per layer


Despite have a higher accuracy in testing data, it is to be noted that the loss of the model is high and in the graph, we can observe that it changes drastically for validation accuracy and loss. 



 
 
 


















5.	Conclusion & Future Works

In conclusion, the coursework demonstrates the usage and strengths of using CNN to classify RPS hand gestures. The dataset was first converted into grayscale and augmented to extrapolate the dataset, then trained the CNN on the training set using various hyperparameters and evaluated its performance on the validation set.

The results showed that the CNN was able to accurately classify RPS hand gestures, achieving a high accuracy rate.

Through this coursework I was able to understand how immensely high potential CNNs has in the field of computer vision and its applications in various industries. Further research can be done to improve the accuracy and efficiency of RPS hand gesture classification using more advanced neural network architectures and techniques.

Overall, this serves as a stepping stone for future developments in the field of RPS gesture recognition and demonstrates the power of machine learning in solving real-world problems.
