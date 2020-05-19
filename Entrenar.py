import sys
import os
import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import  Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import optimizers 
from tensorflow.python.keras import applications

K.clear_session()

data_entrenamiento = './Data/train'
data_validacion = './Data/valid'



"""
Parametros
"""
epocas=20
longitud, altura = 150, 150
batch_size = 32
pasos = 300
validation_steps1 = 200
filtrosConv1 = 32
filtrosConv2 = 64
tamano_filtro1 = (3, 3)
tamano_filtro2 = (2, 2)
tamano_pool = (2, 2)
clases = 3
lr = 0.0001

##Preparamos nuestras imagenes

entrenamiento_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range =20,
    height_shift_range=0.2,
    width_shift_range=0.2,
    featurewise_center=True,
    featurewise_std_normalization=True
    )

test_datagen = ImageDataGenerator(rescale=1. / 255)

# Se utilizan los parametros a los cuales se procesaran las imagenes de la carpeta entrenamiento (Train)
entrenamiento_generador = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')

# Parametros que se utilizaran para el procesamiento de las imagenes de la carpeta Validacion (valid)
validacion_generador = test_datagen.flow_from_directory(
    data_validacion,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')

#Crecion de la Red Neuronal Convolucional
cnn = Sequential()       #Sequential ----> Que estara formada por capas unidas de una en una

# Primera capa.... usamos los parametros para la capa 1
cnn.add(Convolution2D(filtrosConv1, tamano_filtro1, padding ="same", input_shape=(longitud, altura, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

# Siguiente capa... se usa los siguientes parametros
cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding ="same"))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Flatten())

#se agrega una capa en la que definimos las neuronas y la forma de Activacion
cnn.add(Dense(256, activation='relu'))

#Esta linea hace que nuestro sistema use solo el 50% de las neuronas, esto hace que nuestra red aprenda caminos diferentes para clasificar
cnn.add(Dropout(0.5))

#Nuestra ultima capa, y utilizamos Softmax para que la red determine a que clase pertence la imagen que le daremos, esto lo hace por medio de probabilidad
cnn.add(Dense(clases, activation='softmax'))\

#Parametros para optimizar el algoritmo
cnn.compile(loss='categorical_crossentropy',
            optimizer ='Adam',
            metrics=['accuracy','mse'])

#Para el entrenamiento del algoritmo
cnn.fit(
    entrenamiento_generador,
    steps_per_epoch=pasos,
    epochs=epocas,
    validation_data=validacion_generador,
    validation_steps=validation_steps1)


target_dir = './DataModelo/'
if not os.path.exists(target_dir):#si no exite la carpeta DataModelo la crea
  os.mkdir(target_dir)
#Se guardan los archivos modelo.h5 y pesos.h5
cnn.save('./DataModelo/modelo.h5')
cnn.save_weights('./DataModelo/pesos.h5')

