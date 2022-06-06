import os, re, time, json
import PIL.Image, PIL.ImageFont, PIL.ImageDraw
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from matplotlib import pyplot as plt
from datetime import datetime
from keras.models import Model
from keras.callbacks import ModelCheckpoint,EarlyStopping



def plot_acc_loss(history, name):
  acc = result.history['accuracy']
  val_acc = result.history['val_accuracy']
  loss = result.history['loss']
  val_loss = result.history['val_loss']
  plt.figure(figsize=(20, 10))
  plt.subplot(1, 2, 1)
  plt.title("Training and Validation Accuracy")
  plt.plot(acc,color = 'green',label = 'Training Acuracy')
  plt.plot(val_acc,color = 'red',label = 'Validation Accuracy')
  plt.legend(loc='lower right')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.subplot(1, 2, 2)
  plt.title('Training and Validation Loss')
  plt.plot(loss,color = 'blue',label = 'Training Loss')
  plt.plot(val_loss,color = 'purple',label = 'Validation Loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(loc='upper right')
  plt.savefig(name)


def timer(start_time=None):
  #function to track time 
  if not start_time:
      print(datetime.now())
      start_time = datetime.now()
      return start_time
  elif start_time:
      thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
      tmin, tsec = divmod(temp_sec, 60)
      print('Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))




(training_images, training_labels) , (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()



print(training_images.shape,test_images.shape)
def preprocess_image_input(input_images):
  input_images = input_images.astype('float32')
  output_ims = tf.keras.applications.resnet50.preprocess_input(input_images)
  return output_ims


train_X = preprocess_image_input(training_images)
test_X = preprocess_image_input(test_images)

training_images=training_images.reshape((60000,28,28,1))



test_images=test_images.reshape((10000,28,28,1))



def FeatureExtractor(inputs):

  extractor = tf.keras.applications.resnet.ResNet50(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')

  #Modify input layer of ResNet to 1 channel
  cfg = extractor.get_config()
  cfg['layers'][0]['config']['batch_input_shape'] = (None, 224, 224, 1)
  FeatureExtractor= Model.from_config(cfg)(inputs)
  return FeatureExtractor


#Uncomment the dropout layers to try with Dropout 
def classifier(inputs):
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
   #x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
   #x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(10, activation="softmax", name="classification")(x)
    return x


def Model_Final(inputs):

    resize = tf.keras.layers.UpSampling2D(size=(8,8))(inputs)

    resnet = FeatureExtractor(resize)
    classification = classifier(resnet)

    return classification

def compile():
  inputs = tf.keras.layers.Input(shape=(28,28,1))
  
  classification= Model_Final(inputs) 
  model = tf.keras.Model(inputs=inputs, outputs = classification)
 
  model.compile(optimizer='SGD', 
                loss='sparse_categorical_crossentropy',
                metrics = ['accuracy'])
  
  return model


model = compile()

model.summary()

EPOCHS = 100
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

checkpoint = ModelCheckpoint(filepath = 'mRNFMSGDdropout.h5',save_best_only = True,verbose=1)


start_time=timer(None)

result = model.fit(train_X, training_labels, epochs=EPOCHS, validation_split=0.2 ,callbacks = [checkpoint,es], batch_size=64)


plot_acc_loss(result,'mRNFMSGDdropout.png')

print("Evaluating on test data")
results2 = model.evaluate(test_X, test_labels, batch_size=64)
print("test loss, test acc:", results2)

timer(start_time)





