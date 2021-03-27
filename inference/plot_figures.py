import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt



def Plot_model_summary(model):
    "Plots model summary"
    print(model.summary())

def Plot_model_diagram(model, save= True):
    """
    plot model summary diagram
    args:
        model: compiled model
        save: True if model diagram is to be saved as a png else False
    """
    if save:
        keras.utils.plot_model(model, to_file= 'images/CNN_SE_model.png')
    else:
        keras.utils.plot_model(model)

def Plot_learning_curves(history, save_figure = True):
    """
    plot loss curves: train loss & val loss against epochs
    plot accuracy curves: train and val accureacy against epochs
    args:
        history: model.fit data
        save_figure: Boolean
    """
    pass