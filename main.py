import datasets
from data_preprocessing import prepare_directory
from inference import predict_image

import os


# Configurations - to be later moved to config file

data_dir = os.path.join("datasets/GTSRB/Final_Training/Images")
annoations_dir = os.path.join('datasets/GTSRB/annotations')
output_dir = 'datasets/GTRSB/GTRSB_final'
split_ratio = (0.8, 0.1, 0.1)
model_path = 'models/CNN_SE_model.h5'


img_width = 48
img_height = 48
ext = [".ppm"]
batch_size = 32



def main():


    model, history = predict_image.Load_model(model_path)
    predict_image.evaluate_model(model, test_ds)


    print("main function has run")
    return

if __name__ == '__main__':
    main()
