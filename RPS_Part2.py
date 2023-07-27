import tensorflow as tf
import matplotlib.pyplot as plt
loaded_model = tf.keras.saving.load_model("/content/drive/MyDrive/final_modelk5.h5")
loaded_model.summary()
path = '/content/drive/MyDrive/Rock-Paper-Scissors/Rock-Paper-Scissors/test/rock/testrock01-25.png'
im = tf.keras.preprocessing.image.load_img(path)
cim = tf.image.rgb_to_grayscale(im)
im_input = tf.reshape(cim, shape = [1, 300, 300, 1])

predict_proba = sorted(loaded_model.predict(im_input)[0])[2]
predict_class = np.argmax(loaded_model.predict(im_input))

if predict_class == 0:
    predict_label = 'Paper'
elif predict_class == 1:
    predict_label = 'Rock'
else:
    predict_label = 'Scissor'

print('\n')
im_input=tf.reshape(im_input, shape = [300, 300, 1])
plt.imshow(im_input,cmap='gray')
plt.show()
print("\nImage prediction result: ", predict_label)
print("Probability: ", round(predict_proba*100,2), "%")
print('\n')
path = '/content/drive/MyDrive/Rock-Paper-Scissors/Rock-Paper-Scissors/test/paper/testpaper01-15.png'
im = tf.keras.preprocessing.image.load_img(path)
cim = tf.image.rgb_to_grayscale(im)
im_input = tf.reshape(cim, shape = [1, 300, 300, 1])

predict_proba = sorted(loaded_model.predict(im_input)[0])[2]
predict_class = np.argmax(loaded_model.predict(im_input))

if predict_class == 0:
    predict_label = 'Paper'
elif predict_class == 1:
    predict_label = 'Rock'
else:
    predict_label = 'Scissor'

print('\n')
im_input=tf.reshape(im_input, shape = [300, 300, 1])
plt.imshow(im_input,cmap='gray')
plt.show()
print("\nImage prediction result: ", predict_label)
print("Probability: ", round(predict_proba*100,2), "%")
print('\n')
path = '/content/drive/MyDrive/Rock-Paper-Scissors/Rock-Paper-Scissors/test/scissors/testscissors01-10.png'
im = tf.keras.preprocessing.image.load_img(path)
cim = tf.image.rgb_to_grayscale(im)
im_input = tf.reshape(cim, shape = [1, 300, 300, 1])

predict_proba = sorted(loaded_model.predict(im_input)[0])[2]
predict_class = np.argmax(loaded_model.predict(im_input))

if predict_class == 0:
    predict_label = 'Paper'
elif predict_class == 1:
    predict_label = 'Rock'
else:
    predict_label = 'Scissor'

print('\n')
im_input=tf.reshape(im_input, shape = [300, 300, 1])
plt.imshow(im_input,cmap='gray')
plt.show()
print("\nImage prediction result: ", predict_label)
print("Probability: ", round(predict_proba*100,2), "%")
print('\n')
