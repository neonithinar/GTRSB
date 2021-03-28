import tensorflow as tf
import numpy as np
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import os

img_width = 48
img_height = 48
IMAGES_PATH = 'images/'
CLASS_NAMES = ['00000_speed_limit_20kmph', '00001_speed_limit_30_kmph', '00002_speed_limit_50_kmph',
               '00003_speed_limit_60_kmph', '00004_speed_limit_70_kmph', '00005_speed_limit_80_kmph',
               '00006_end_of_speed_limit', '00007_speed_limit_100_kmph', '00008_speed_limit_120_kmph',
               '00009_no_passing', '00010_no_passing_for_vehicles', '00011_right_of_way_at_the_intersection',
               '00012_priority_road', '00013_yeild', '00014_stop', '00015_no_vehicles', '00016_vehicles_over_34_metres',
               '00017_no_entry', '00018_general_caution', '00019_dangerous_curve_to_left', '00020_dangerous_curve_to_right',
               '00021_double_curve', '00022_bumpy_road', '00023_slippery_road', '00024_roads_narrows_on_the_right',
               '00025_road_work', '00026_traffic_signals', '00027_pedestrians', '00028_children_crossing',
               '00029_bicycle_crossing', '00030_beware_of_ice_or_snow', '00031_wild_animals_crossing',
               '00032_end_of_all_speed_and_passing', '00033_turn_right_ahead', '00034_turn_left_ahead',
               '00035_ahead_only', '00036_go_straight_or_right', '00037_go_straight_or_left',
               '00038_keep_right', '00039_keep_left', '00040_roundabout_mandatory', '00041_end_of_passing',
               '00042_end_of_no_passing_by']

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
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
    pd.DataFrame(history).plot(figsize = (8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
    plt.show()
    if save_figure:
        save_fig(fig_id='Learning_curves')

def Preproces_image(image_path):
    test_image = keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))
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
    prediction = CLASS_NAMES[np.argmax(predictions)]
    if show_img:
        plt.imshow(test_image)
        plt.title(prediction)
        plt.axis('off')
        plt.show()
        return print(f'Predictiona: {prediction}')
    else:
        return print(f'Predictiona: {prediction}')

