import tensorflow as tf
from tensorflow import keras
import numpy as np
import os


def Load_model(model_path):
    """
    Loads the keras model.h5 for inference
    Else initiates Training of model

    args:
        model_dir: path to models folder

    returns:
        Trained model.h5 for inference

    """
    print("Loading model")
    if os.path.exists(model_path):
        model = keras.models.load_model(model_path)

    else:
        print('no model found')
        

# TODO: create inference function with model.predict
