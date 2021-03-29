import tensorflow as tf
import numpy as np
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from configurations.configs import Configurations
import os

Configs = Configurations()


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(Configs.IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

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
        keras.utils.plot_model(model, to_file= "images/CNN_SE_model.png")
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
    pd.DataFrame(history).plot(figsize = (8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
    if save_figure:
        save_fig(fig_id="Learning_curves")
    plt.show()


def Preproces_image(image_path):
    test_image = keras.preprocessing.image.load_img(image_path, target_size=(Configs.img_height, Configs.img_width))
    image_array = keras.preprocessing.image.img_to_array(test_image)
    img_batch = tf.expand_dims(image_array, axis=0)

    return test_image, img_batch



def Predict_image(image_path, model, show_img = True):

    """
    Predicts the traffic image using model
    args:
        image_path: path to image
    """
    test_image, preprocessed_image = Preproces_image(image_path)
    predictions = model.predict(preprocessed_image)
    prediction = Configs.CLASS_NAMES[np.argmax(predictions)]
    if show_img:
        plt.imshow(test_image)
        plt.title(prediction)
        plt.axis("off")
        plt.show()
        return print(f"Predictiona: {prediction}")
    else:
        return print(f"Predictiona: {prediction}")

