# music-genre-network-project
Author: Dagny Brand

## Part 0
This repository holds the Semester Project for CSE 40868. For the project, I have chosen to do the Music Genre Classification pre-defined project topic.

This project will utilize an Anaconda virtual environment to run PyTorch.

## Part 1

This network is intended to classify the musical genre of input audio samples. For the brainstorming aspect of this project, I will assume my network will take in calculated audio values instead of a song itself. I started by doing research on how humans can tell the difference between different musical genres. [Acousterr](https://www.acousterr.com/blog/how-to-identify-different-genres-of-music) provided me with lots of explanation on how to identify the difference. There are at least three major elements that I (or my classifier) should watch for.

#### Types of Instruments

Rock: electronic guitar, electric bass, drums, singing, extensive use of snare drum

Jazz: horn, keyboard, vocals, guitars, drums

Pop: singing, electronic sounds

Country: banjos, electric guitars, acoustic guitars, steel guitars, fiddles, harmonicas

Blues: guitar, piano, harmonica, bass, drums, blues harp, slide guitar, xylophone


 #### Key Rhythm Features

 Rock: song based, verse-chorus form

 Jazz: swing and blue notes, call and response vocals, polyrhythms

 Pop: emphasis on recording and production, simple melody, verse-chorus form

 Country: simple folk tunes, simple forms

 Blues: lyrics, basae line, instrumentation



 #### Notable Timing

 Rock: 4/4 time signature

 Jazz: Improved

 Pop: thirty-two-bar form

 Country: includes slow and fast songs

 Blues: 12-bar blues chord


 I need to learn how these features translate to recognizable data for a network to learn from. After exploring the [reference GitHub](https://github.com/mdeff/fma) and dataset as well as consulting ChatGPT on which musical features are needed to classify genres, there are a few features I think are needed to extract in order to classify audio. These include **mel-frequency cepstral coefficients (MFCCs)** (coefficients that represent the power spectrum of a sound -- can help identify certain instruments), a **chromogram** (energy distribution of pitch classes -- can help identify instruments), **zero-crossing rate** (rate at which signal changes sign -- helps identify noisiness or percussiveness), **spectral features** (such as the centroid, contrast, roll-off, and bandwidth -- provide information about the distribution of energy in different frequency bands), **tonnetz** (temporal features -- capture overall trends), and the **root mean square** (captures energy or loudness). I can use the [librosa](https://librosa.org/doc/latest/feature.html) library to extract these features from my own input songs, as was done in the reference GitHub to compile its dataset. 


 To create this classifier, I need three data subsets: one for training, one for validation, and one for final testing. To create the training and validation datasets, I will  use scikit's test_train_split to split the dataset from the reference GitHub  randomly into two seperate training and validation sets. I am currently thinking of using 70% of the data for training and 30% for validation, but if the classifier seems to need more data for training to recognize the patterns more accurately, this number can be adjusted. For the final testing set, I will create a new data set using my own song choices and measurements from the librosa library, as this data will be completely new for the network. One thing I do need to be aware of is if I pick a song that is already in the reference GitHub dataset. 


 I do not have a strong background in neural networks but I am very excited for this course and for this project! My idea for this network is to create a multi-level perceptron network. I need a network that can take in a vector of coefficients (like the MFCC value, zero-crossing rate, etc) and output a classification, like a 1 for the rock genre. Another idea, which I discussed with Adam, is using the spectrograms of the audio files. A convolutional neural network, which can analyze images, can take in an image of a spectrogram and use that to produce a genre classification. I think this sounds super interesting as well!

 I am excited to learn more about designing classifiers to determine how many layers I should have, what kind of layers I should have, what activation functions I should use, and more information about how to design the network.


## Part 2

For this project, I have found three subsets of data, training, validation, and testing, to create my dataset. 

#### Training and Validation Data
In Part 1, I discussed two possible routes for this project, one that focused on features and one that focused on spectrograms (images). I have decided to take the spectrogram route and to create a CNN that takes a spectrogram image as input and outputs the genre classification. As such, the dataset from the original reference GitHub won't work. Instead, I have downloaded a Kaggle dataset that contains spectrograms of the GTZAN Dataset. The dataset is called GTZAN Dataset - Music Genre Classification, and the Kaggle page is linked [here](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification?resource=download). The Kaggle datasets are open source.

The original GTZAN Dataset is a public dataset originally from "Musical genre classification of audio signals" by G. Tzanetakis and P. Cook. This paper is linked [here](https://ieeexplore.ieee.org/document/1021072). The GTZAN dataset is the most used public dataset for machine learning for music genre classification. This dataset contains 1000 audio files, each 30 seconds in length. These files are divided into 10 different genres, each with 100 corresponding audio files. The dataset I downloaded includes these audio files as spectrograms, which were created using the librosa library. So, in total, my complete dataset includes 100 spectrograms for each of the 10 genres I will be able to classify: blues, classical, country, disco, hiphop, jazz (only has 99 files), metal, pop, reggae, and rock. This gives me a total of 999 spectrograms. Here are a few spectrograms from my dataset:

*Classification: Blues*

![Blues Spectrogram](/data_samples/blues00009.png "Classification: Blues")

*Classification: Hip Hop*

![Hip Hop Spectrogram](/data_samples/hiphop00003.png "Classification: Hip Hop")

*Classification: Metal*

![Metal Spectrogram](/data_samples/metal00002.png "Classification: Metal")


I compiled the images, which were originally split into 10 seperate folders for each genre, into one csv file containing the image name and the correct genre classification to use as input in my neural network for both training and validation. Then, I used scikit test_train_split to split my csv entries randomly into 70% training data and 30% validation data. The small Python code I used to do this is located in the [split_data.py](split_data.py) file in this GitHub.

#### Unknown Testing Data

For the "unknown" testing dataset, I ran 20 of my own song choices, 2 per genre as classified by Apple Music, through a [Spectrogram Creator](https://convert.ing-now.com/audio-spectrogram-creator/). Here is an example entry of the testing dataset:

*Back in Black by ACDC (Rock)*

![Back in Black Spectrogram](/test_samples/back_in_black_(rock).png)


## Part 3

### Testing the First Solution on a Single Validation Sample
To test my first solution model on a single validation sample, run [test_model.py](test_model.py). This program requires the torch, torchvision, sklearn, cv2, and numpy packages. It also uses the CNN class from the [model_architecture.py](model_architecture.py) file. The model weights are located in [first_solution.pth](first_solution.pth) and a single validation sample is the spectrograph of a Country song seen in [validation_sample/country00044.png](validation_sample/country00044.png). If you run [test_model.py](test_model.py) from a clone of this GitHub repository with the necessary packages, it should run without error. If you run the file from a different folder, you may need to redefine the links to the model weights and the validation sample within [test_model.py](test_model.py). The program will output "The song genre is Country" as the single validation input is a Country song.

### First Solution Report
#### Neural Network Architecture
My model uses a Convolutional Neural Network architecture, as defined in [model_architecture.py](model_architecture.py). The first convolutional layer has a kernel size of (7, 7) and a stride of (6, 6). It uses a large kernel and stride to minimize the size of the input image and allow for a faster and more efficient training and prediction time. The first layer has 64 output channels, the second layer has 128 output channels, and the third layer has 64 output channels. These output channels help the neural network to examine many different features of the spectrographs. The neural network utilizes a 2x2 maxpool to reduce the number of parameters needed in the network and shrink its size and training time. The neural network also uses the ReLU activation function after each convolutional layer to introduce nonlinearity into the network, as ReLU is often used in CNNs. During training, which was done with Google Colab and seen in [Semester_Project.ipynb](Semester_Project.ipynb), the model uses the Adam optimizer with a learning rate of 0.001. I first tried the SGD optimizer, which had shown to improve accuracy during the Class Practicals. However, the accuracy of the network was only around 20% when SGD was used, no matter the learning rate. Using Adam for the optimizer significantly increased the accuracy, so I switch to using Adam. Lastly, the model uses the Cross Entropy Loss loss function, which is good for classification problems and which we have used in Class Practicals.

#### Classification Accuracy
The classification accuracy is tested at the bottom of [Semester_Project.ipynb](Semester_Project.ipynb). The classification accuracy on the validation data set is 59.667%. The classification accuracy on the training data set is 85.122%.

#### Commentary and Improvements for Final Solution
I think the model did fairly well for the first solution. It was trained on 50 epochs and reached almost 60% accuracy on the validation data set, which is much better than random guessing. The model did have a 15% higher accuracy when tested on the training data set compared to the validation data set. This means that there is possibly overfitting within the model. To decrease this, I will try increasing the size of the training data set for the final solution to allow the model to see more diverse data. To improve the final solution, I will also examine if there is a trend in which genres are being misclassified. It is possible that by randomly splitting the data into training and validation sets, the training set ended up having many more samples of one type of data. By increasing the size of the training set, I can try to reduce this error. I can also try using different optimizers. Changing the optimizer from SGD to Adam had a significant impact on the model, so it is possible that using a different optimizer, such as AdaGrad, will improve it further. I don't necessarily want to add more layers or parameters to my model because making the model deeper will not necessarily improve accuracy, but it will increase the size of the model. The model is currently about 52 MB, so I may even experiment with shrinking the number of parameters used in the model, and thus, the size used. I could also decrease the maximum number of channels used from 128 to 64, as it is possible that the number of significant features is much smaller than the first solution assumes. Lastly, I might try changing the way that accuracy is calculated or using other measures of accuracy to successfully evaluate the models strengths and weaknesses. 


## Part 4

### Testing the Final Solution on a Single Test Sample
To test my first solution model on a single test sample, run [test_model.py](test_model.py). This program requires the torch, torchvision, sklearn, cv2, and numpy packages. It also uses the CNN class from the [model_architecture.py](model_architecture.py) file. The model weights are located in [final_solution.pth](final_solution.pth) and a single test sample is the spectrogram of the song At Last by Etta James, which is a Blues song, seen in [test_samples/at_last_(blues).png](test_samples/at_last_(blues).png). If you run [test_model.py](test_model.py) from a clone of this GitHub repository with the necessary packages, it should run without error. If you run the file from a different folder, you may need to redefine the links to the model weights and the validation sample within [test_model.py](test_model.py). The program will output "The song genre is Blues" as the single test input is a Blues song.


### Final Solution Report
#### Gathering Test Dataset
To create my "unknown" test dataset, I gathered the .mp4 files for 20 songs from Apple's iTunes. I found two songs per genre, as classified by Apple Music. I then used a [Spectrogram Creator](https://convert.ing-now.com/audio-spectrogram-creator/) to create spectrograms of the songs, using an output size of 1024x1024, a color level of 1, an intensity level of 1, and a density level of 1 to best match the style of the training data spectrograms. I used Preview to reformat the images to a size of 432x288, which is the input size for my neural network. An example of these spectrograms is seen in [Part 1](README.md#unknown-testing-data) of this readme. These test spectrograms were not necessarily made with the same program or specifications as the training and validation spectrograms. I think this dataset is sufficient to test my neural network's capabilities for a few reasons. First, the original GTZAN dataset was published in 2002. My dataset includes songs from after 2002, making it a more modern representation of the songs a typical modern user would enter into the model. So, this new songs help to test the network's abilities in a more general sense. The music industry is constantly changing and many modern songs do not fit perfectly into one genre or another. The testing dataset helps to see how the network handles songs that cannot be easily categorized into a genre. Secondly, the training and validation datasets include spectrograms made from 30 second snippets of songs. On the other hand, the testing set includes spectrograms made from complete songs, ranging from 3 to 5 minutes. In this way, the testing set can test how the network responds to spectrograms made from different sized audio clips, as well as to complete songs, which may contain many different types of tunes and not fit perfectly into one single genre. Lastly, it is not specified which spectrogram-creating tool the GTZAN dataset was made with. It is likely that the testing set spectrogeams were created with a different program than the training and validation sets, although the spectrograms look similar and standardized. Because of this, the testing set is able to test how the network responds to spectrograms that may be slightly different than the training and validation spectrograms, sufficiently testing the networks generalization capabilities.  


### Test Set Accuracy
The final solution is a smaller network than the first solution. Everything remained the same as explained in [Part 3](README.md#neural-network-architecture) except for the channel input and output numbers for each convolutional layer and the optimizer function. The first layer now has 16 output channels, the second layer has 32 output channels, and the third layer has 16 output channels. The optimizer function is an Adagrad optimizer with a learning rate of 0.01. The final solution has an accuracy of 62.67% when tested on the validation set and an accuracy of 30.00% when tested on my "unknown" test set. 

### Commentary on Test Set
The network was able to classify the Hip Hop, Blues, and Classical genres perfectly but thought that the rest of the genres fell into these three categories as well. I think one of the main reasons that the network performed worse on the testing set is because the spectrograms in the testing set were made from complete songs instead of 30 second snippets. Here is a side by side comparison of a Hip Hop song from the testing set (Thrift Shop by Macklemore) and a Hip Hop song from the validation set, respectively. The network was able to correctly classify both of these images.

![Testing](/test_samples/thrift_shop_(hiphop).png) ![Training](/data_samples/hiphop00018.png)

The similarities between the two hip hop spectrograms are clear. The tall spikes are slightly spread out, there is high yellow density at the bottom, the spikes are all pretty tall, and there are black spaces within both spectrograms.


Here is a side by side comparison of a Country song from the testing set (Cruise by Florida Georgia Line) and a Country song from the validation set, respectively. The network was able to classify the training sample as Country, but it classified the testing sample as Hip Hop. 

![Testing](/test_samples/cruise_(country).png) ![Training](/data_samples/country00007.png)

The training sample for Country is clearly different from the training sample for Hip Hop. Despite both samples having tall spikes and a high density of yellow toward the bottom of the spectrogram, the spaces and lines in the Country sample are much more jagged than those in the Hip Hop sample. However, because the testing samples are slightly less zoomed-in, the differences between the Country testing sample and the Hip Hop testing sample are not as prominent. While the human eye can see the bigger yellow intense sections at the bottom of the Country sample, this difference is not as obvious as the jagged lines in the training sample. This shows that because the testing samples are of larger time frames, resulting in them being more zoomed-out than the training samples, the differences between genres are not as prominent. Because the neural network was trained soley on 30 second audio clips, the network is unable to successfully distiguish between some genres when the song clips are longer, although it is able to successfully classify Blues, Hip Hop, and Classical songs. To fix this issue, I would train the network on longer audio clips to allow it to learn how spectrograms of complete songs look and differ from 30 second clips.   



