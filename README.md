# DeepLearningPictureDogs
Classify pictures with Deep Learning Model

This study is to learn to use, create and train  deep learning CNN models (ResNet / VGG).

An association which helps and hosts a lot of dogs need to classify pictures of their animals.
To do that, it is possible to use CNN models by training on Stanford Dogs Dataset (http://vision.stanford.edu/aditya86/ImageNetDogs/).

These is my works about this study : 

01_OC_IML_Project_6_explorations.ipynb : exploration

02_OC_IML_Project_6_my_neural_network.ipynb & 03_OC_IML_Project_6_my_neural_network_2.ipynb : Creation / test of a neural network  CNN from scratch.

04_OC_IML_Project_6_10breeds.ipynb & 05_OC_IML_Project_6_3breeds.ipynb : VGG-16 Neural Network transfert learning 

06_OC_IML_Project_6_3breeds_ResNet50.ipynb & 07_OC_IML_Project_6_10breeds_ResNet50.ipynb : ResNet-50 transfert learning

08_OC_IML_Project_6_my_neural_network_DAugm_01.ipynb : data augmentation on my neural network from scratch


09_OC_IML_Project_6_10breeds_ResNet50_script.ipynb : notebook to create and test script
- script_predict_breed.py : command line script to predict breed (ResNet50 10 breeds)
	- ResNet50_TL_10b_9.h5 : transfert learning model
	- df_dogs.pkl : contains  list of breeds (and other stuff about training pictures)

