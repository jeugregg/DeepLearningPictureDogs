# -*-coding:utf-8 -*

'''
  Command line Script to predict dog breed from an image path

  How to use : 

  - Just type : python script_predict_breed.py your_img_path 

              - return the breed of your image your_img_path.

  - exemple : python script_predict_breed.py AH_n02088094_185.jpg
'''
def predict_breed(path_img, path_mdl="ResNet50_TL_10b_9.h5"):
  # import useful libs
  import os
  import pickle
  from sklearn.externals import joblib
  import numpy as np
  import pandas as pd
  # keras
  import keras
  from keras.models import load_model
  # keras already trained
  from keras.applications.resnet50 import ResNet50
  # keras image preparation
  from keras.applications.resnet50 import preprocess_input, decode_predictions
  from keras.preprocessing.image import load_img, img_to_array
  from keras.models import Model

  def predict_breed_from_one_path(path_img, mdl):
    '''
    Predict Breed Name from one image path

    example : breedName = predict_breed_from_one_path(path_img, model)
    '''
    def find_breed(y, dict_breed):
      '''
      Find Breed Name from y output of CNN model
      breed_name = find_breed(y, dict_breed)
      '''
      return dict_breed[np.argmax(y)]

    def import_breeds():
    # find breed name into data dog use to train model
      path_df_dogs = os.getcwd() + '/df_dogs.pkl'
      df_dogs = joblib.load(path_df_dogs)
      dict_breed = dict()
      for id_class in df_dogs["class"].unique(): 
        dict_breed[id_class] = df_dogs[df_dogs["class"] == \
                                      id_class]["breed"].values[0]
      return dict_breed
  
    def load_prepare_img(path_img):
      '''
      Load image from path and prepare for VGG-16
      return : 
        img : np array
        img_raw : Image from PIL

      example : img, img_raw = load_prepare_img(path_img)
      '''

      img_raw = load_img(path_img, target_size=(224, 224))  # Charger l'image
      img = img_to_array(img_raw)  # Convertir en tableau numpy
      img = img[np.newaxis, :]
      img = preprocess_input(img)  # Prétraiter l'image comme le veut VGG-16
      img_raw
      return img, img_raw

    dict_breed = import_breeds()

    print(path_img)
    img, img_raw = load_prepare_img(path_img)
    y = mdl.predict(img) 
    if y.shape[1] >= 1000:
      breedName = decode_predictions(y, top=3)[0]
      print('Model : Top 3 :', breedName)
    else:
      print("Probabilities : ", y)
      breedName = find_breed(y, dict_breed)
    print(breedName)
    return breedName, img_raw

  # re-load model
  mdl = load_model(path_mdl)

  # predict from image path
  breedName, img_raw = predict_breed_from_one_path(path_img, mdl)

  # return thee breed 
  return breedName

if __name__ == "__main__":
  import sys
  predict_breed(sys.argv[1])