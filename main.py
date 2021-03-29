import datasets
from configurations.configs import Configurations
from data_preprocessing import prepare_directory
from inference.evaluation import Evaluate_model, Load_model
from inference.plot_figures import Plot_model_summary, Plot_model_diagram, Plot_learning_curves, Predict_image
from training_and_execution import train_model
# from data_preprocessing import preprocess_data

import os


Configs = Configurations()


def main():
    y = input("Do you want to train your own model? press y/n...")
    if y == 'y' or y == 'Y':
        train_model.Train()
    elif y == 'n' or y == 'N':
        print('Looking for stored model file')

    else:
        print("invalid input")
    model, history = Load_model(Configs.model_path)
    print("Evaluating saved model with test set")
    Evaluate_model(model)
    print("Plotting model summary and model diagram")
    Plot_model_summary(model)
    # Plot_model_diagram(model, save= True) # model diagram not supported in tf.__version__ 2.3.0
    # but supported in version 2.4.1
    print("plotting learining curves")
    Plot_learning_curves(history)
    print("INFERENCE: TRYING TO PREDICT AN UNKNOWN TRAFFIC IMAGE FROM WEB \n "
          "If you want to change the test file, please change the test file path in config.py file")
    Predict_image(Configs.test_image_path, model, show_img = True)





    print("main function has run")
    return

if __name__ == '__main__':
    main()
