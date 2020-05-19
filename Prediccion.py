import numpy as np
import os
from os.path import isfile, join
from keras_preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

#Parametros longitud y altura deben ser iguales a los establecidos en el script de entrenar.py
longitud, altura = 150, 150
modelo = './DataModelo/modelo.h5'
pesos_modelo = './DataModelo/pesos.h5'

#Cargamos el modelo y los pesos a la variable cnn
cnn = load_model(modelo)
cnn.load_weights(pesos_modelo)

#creacion de la funcion predict
def predict(file):
  #se carga la imagen
  x = load_img(file, target_size=(longitud, altura))
  # La convertimos en arreglo
  x = img_to_array(x)
  # Le añadimos una dimension mas para que podamos procesar nuestra informacion
  x = np.expand_dims(x, axis=0)
  # hacemos la prediccion sobre nuestra imagen y la almacenara en un arreglo
  array = cnn.predict(x) #[[0,0,1]]
  # se usa [0] porque el arreglo es de dos dimensiones y nomas nos importa la primera
  result = array[0] 
  # Esto nos dara la posicion del 1
  answer = np.argmax(result)
  print(answer)
  if answer == 0:
    print("ES MANZANA")
  elif answer == 1:
    print("ES BANANA")
  elif answer == 2:
    print("ES NARANJA")
  return answer

#Se envian las imagenes desde una carpeta
contenido= os.listdir('Imagenes de prueba')

# Se recorre las imagenes una por una teniendo cada una su prediccion
for Imagen in contenido:
  print (Imagen)
  predict(Imagen)





