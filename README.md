
![Banner](https://github.com/neonithinar/GTRSB/blob/main/images/Banner.png)
<!-- retro visitor counter -->
<p align="center"> 
  <img src="https://profile-counter.glitch.me/{neonithinar}/count.svg" />
</p>

<!-- Welcome Message -->
<h1>GTSRB CNN classifier with custom Squeeze and Excitation Block</h1>

<h3>Project files to classify </h3>



**How to Use**  

* Clone this repo
* Run _server.py_ from your terminal
* Copy the url in the output in your browser to run the web app locally
* Give a language string you need to identify in the input field
* Check the obtained Result !


**How this works ??** <img src="https://media.giphy.com/media/TKXabSgn2ouV8vmTue/giphy.gif" width="40px">

* The string given to the input form is passed on to a pretrained Neural network model (model_1.h5) after cleaning. 
* The model outputs a set of probabilities from which the most probable language is returned to the web application as the prediction
* **[Checkout my 5 min discription of the project on youtube here... ! ](https://youtu.be/PBPjTO-yTdQ)**

**N-gram based Language Detection Model**

* An N-gram is an N-character slice of a longer string. Although in the literature the term can include the notion of any co-occurring set of characters in a string (e.g., an N-gram made up of the first and third character of a word), in this
project we use the term for contiguous slices only. Typically, we will be using <h5>character trigrams<h5> 
* eg; the word ```TEXT``` will become ```_TE, TEX, EXT, XT_, T_ _``` (begining and end of the word is padded with white space)
* The idea is to identify *Character Trigrams* unique to each language and use them to classify the languages
* The model was trained on a subset of [Tatoeba dataset](https://downloads.tatoeba.org/exports/sentences.csv) which contains over 300 languages. 
* Currently the model is able to identify dutch, finnish, polish, lituanian, czech, swedish, arabic, macedonian, danish, and serbian (10 languages. But will soon be updated to identify more languages)
* After cleaning the dataset, The most common *Language Trigrams* were extracted from each of the languages, and a vocabulary for each language was created from the training set based on which the whole training, validation and test set were transformed using ```sklearn.CountVectorizer ``` 
	
* These extracted features were fed to a simple feedforward Neural network with 3 hidden layers to classify the languages.
* ```Early stopping``` and ```model checkpoint```  callbacks to prevent overfitting of the model


**References**

* _William B. Cavnar et. al._ , [N-Gram-Based Text Categorization](https://www.researchgate.net/publication/2375544_N-Gram-Based_Text_Categorization) 
* _A. Simões, J.J. Almeida, and S.D. Byers_, [Language identification: a neural network approach. (2014)](https://www.researchgate.net/publication/290102620_Language_identification_A_neural_network_approach)

I would very much like to extend my gratitude to user _@BobVonBob_ in discord (code bullet and co server, Machine Learning Channel) for suggesting the use of bigram frequencies to identify language. Which later led me to read The paper on N-gram based Text recognition (William B. Cavnar et.al) and other useful literatures
