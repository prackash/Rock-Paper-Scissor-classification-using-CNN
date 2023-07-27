import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
loaded_model = tf.keras.saving.load_model("/content/drive/MyDrive/final_modelk5.h5")
def rps_game(aim_input,bim_input):
  
  a=aim_input
  b=bim_input
  w=a
  apred= np.argmax(loaded_model.predict(a))
  bpred = np.argmax(loaded_model.predict(b))
  print(apred)
  print(bpred)
  if (apred == 0 and bpred ==0):
    print("Game is a tie")
  elif apred == 0 and bpred ==1:
    print("Player a has won")
    w=tf.reshape(a, shape = [300, 300, 1])
    plt.imshow(w,cmap='gray')
    plt.show()
  elif apred == 0 and bpred ==2:
    print("Player b has won")
    w=tf.reshape(b, shape = [300, 300, 1])
    plt.imshow(w,cmap='gray')
    plt.show()

  elif apred == 1 and bpred == 0:
    print("Player b has won")
    w=tf.reshape(b, shape = [300, 300, 1])
    plt.imshow(w,cmap='gray')
    plt.show()
  elif apred == 1 and bpred == 1:
    print("Game is a tie")
  elif apred == 1 and bpred == 2:
    print("Player a has won")
    w=tf.reshape(a, shape = [300, 300, 1])
    plt.imshow(w,cmap='gray')
    plt.show()

  elif apred == 2 and bpred == 0:
    print("Player a has won")
    w=tf.reshape(a, shape = [300, 300, 1])
    plt.imshow(w,cmap='gray')
    plt.show()
  elif apred == 2 and bpred == 1:
    print("Player b has won")
    w=tf.reshape(w, shape = [300, 300, 1])
    plt.imshow(w,cmap='gray')
    plt.show()
  elif apred == 2 and bpred == 2:
    print("Game is a tie")
apath = '/content/drive/MyDrive/Rock-Paper-Scissors/Rock-Paper-Scissors/test/paper/testpaper01-15.png'
aim = tf.keras.preprocessing.image.load_img(apath)
acim = tf.image.rgb_to_grayscale(aim)
aim_input = tf.reshape(acim, shape = [1, 300, 300, 1])

bpath = '/content/drive/MyDrive/Rock-Paper-Scissors/Rock-Paper-Scissors/test/rock/testrock01-25.png'
bim = tf.keras.preprocessing.image.load_img(bpath)
bcim = tf.image.rgb_to_grayscale(bim)
bim_input = tf.reshape(bcim, shape = [1, 300, 300, 1])

rps_game(aim_input,bim_input)

