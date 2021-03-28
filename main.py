import datasets
from configurations.configs import Configurations
from data_preprocessing import prepare_directory
from inference.evaluation import Evaluate_model, Load_model
from inference.plot_figures import Plot_model_summary, Plot_model_diagram, Plot_learning_curves, Predict_image
# from data_preprocessing import preprocess_data

import os


Configs = Configurations()


def main():

    model, history = Load_model(Configs.model_path)

    Evaluate_model(model)
    Plot_model_summary(model)
    # Plot_model_diagram(model, save= True)
    Plot_learning_curves(history)
    Predict_image(test_image_path, model, show_img = True)





    print("main function has run")
    return

if __name__ == '__main__':
    main()
