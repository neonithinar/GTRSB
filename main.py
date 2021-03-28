import datasets
from data_preprocessing import prepare_directory
from inference.evaluation import Evaluate_model, Load_model
from inference.plot_figures import Plot_model_summary, Plot_model_diagram, Plot_learning_curves, Predict_image
# from data_preprocessing import preprocess_data

import os


# Configurations - to be later moved to config file

data_dir = os.path.join("datasets/GTSRB/Final_Training/Images")
annotations_dir = os.path.join('datasets/GTSRB/annotations')
output_dir = 'datasets/GTRSB/GTRSB_final'
split_ratio = (0.8, 0.1, 0.1)
model_path = 'models/CNN_SE_model.h5'
test_image_path ='images/test_img_1.jpg'


img_width = 48
img_height = 48
ext = [".ppm"]
batch_size = 32



def main():

    model, history = Load_model(model_path)

    Evaluate_model(model)
    Plot_model_summary(model)
    # Plot_model_diagram(model, save= True)
    Plot_learning_curves(history)
    Predict_image(test_image_path, model, show_img = True)





    print("main function has run")
    return

if __name__ == '__main__':
    main()
