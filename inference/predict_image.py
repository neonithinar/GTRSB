import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import train_model
import pickle

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
        with open('models/history.pkl.pkl', 'rb') as f:
            history = pickle.load(f)

    else:
        print('no model found, preprocessing data and training model')
        train_ds, val_ds, _ = preprocess_data.Get_datasets()
        train_model.Train(train_ds, val_ds)
        model = keras.models.load_model(model_path)
        with open('models/history.pkl.pkl', 'rb') as f:
            history = pickle.load(f)

    return model, history

# TODO: create inference function with model.predict


def evaluate_model(model):
    """
    Evaluates the model against a given test_set

    """
    _, _, test_ds = preprocess_data.Get_datasets()
    return print(model.evaluate(test_set))
