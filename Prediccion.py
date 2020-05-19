import numpy as np
import os
from os.path import isfile, join
from keras_preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

longitud, altura = 150, 150
modelo = './DataModelo/modelo.h5'
pesos_modelo = './DataModelo/pesos.h5'
cnn = load_model(modelo)
cnn.load_weights(pesos_modelo)

def predict(file):
  x = load_img(file, target_size=(longitud, altura))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = cnn.predict(x)
  print(array)
  result = array[0]
  print(result)
  answer = np.argmax(result)
  print(answer)
  if answer == 0:
    print("ES MANZANA")
  elif answer == 1:
    print("ES BANANA")
  elif answer == 2:
    print("ES NARANJA")
  return answer


contenido= os.listdir(r'C:\Users\maan9\OneDrive\Documents\Red_Neuronalv2\Imagenes de prueba')
print(contenido)
for Imagen in contenido:
  print (Imagen)
  predict(Imagen)





