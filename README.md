
![Banner](https://github.com/neonithinar/GTRSB/blob/main/images/Banner.png)
<!-- retro visitor counter -->
<!-- <p align="center"> 
  <img src="https://profile-counter.glitch.me/{neonithinar}/count.svg" />
</p> -->

<!-- Welcome Message -->
<h1>GTSRB CNN classifier with custom Squeeze and Excitation Block</h1>

<h3>Project files to classify Traffic signs to 43 diferent classes</h3>



**How to setup the environment ?**  

* Clone this repo
* install the libraries mentioned in the requirements.txt or use a conda env to install the exact env in your system
* OR install the following libraries manually
	* tensorflow GPU enabled version for your system
	* numpy
	* pandas
	* matplotlib
	* opencv
	* **[splitfolders](https://files.pythonhosted.org/packages/b8/5f/3c2b2f7ea5e047c8cdc3bb00ae582c5438fcdbbedcc23b3cc1c2c7aae642/split_folders-0.4.3-py3-none-any.whl)**  ($ pip install splitfolders)



**How to Use the project ??** 

* run main.py in your virtual environment
* If GTSRB dataset was not found in the dataset folder, the file will automatically download the dataset and continue with the program
* in the prompt that follows, type "y" if you want to train a fresh model or "n" if you want to use a pretrained network

* If you select "y" the program will proceed with a fresh trainig of the model and save weights and history of the current model with current timestamp in the models folder.


* If you type "n" the model will proceed to evaluate the test set. further more it will also plot the learning curves and make prediction on the test file.

<h4> Making changes <h4>

All training and model parameters are listed in the *configurations/configs.py* file.If you need to change any parameters to the training, you can edit the variables in the Configurations class. 

** Making a custom prediction**
* To make a custom prediction, move the image to predict into the images folder. copy the image path into test_image_path in the Configurations class within configs.py file.
* Run the main file to get the prediction of the image

**Model Architecture**
<p align="center"> 
  <img src="https://github.com/neonithinar/GTRSB/blob/main/images/CNN_SE_model.png" />
</p>


**References**

* J. Stallkamp, M. Schlipsing, J. Salmen, and C. Igel. The German Traffic Sign Recognition Benchmark: A multi-class classification    competition. In Proceedings of the IEEE International Joint Conference on Neural Networks, pages 1453–1460. 2011.
* Hu, Jie, Li Shen, and Gang Sun. "Squeeze-and-excitation networks." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 7132-7141. 2018.


