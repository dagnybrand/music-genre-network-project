# music-genre-network-project
Author: Dagny Brand

## Part 0
This repository holds the Semester Project for CSE 40868. For the project, I have chosen to do the Music Genre Classification pre-defined project topic.

This project will (hopefully, if all goes well) utilize an Anaconda virtual environment to run PyTorch.

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

In Part 1, I discussed two possible routes for this project, one that focused on features and one that focused on spectrograms (images). I have decided to take the spectrogram route and to create a CNN that takes a spectrogram image as input and outputs the genre classification. As such, the dataset from the original reference GitHub won't work. Instead, I have downloaded a Kaggle dataset that contains spectrograms of the GTZAN Dataset. The dataset is called GTZAN Dataset - Music Genre Classification, and the Kaggle page is linked [here](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification?resource=download). The Kaggle datasets are open source.

The original GTZAN Dataset is a public dataset originally from "Musical genre classification of audio signals" by G. Tzanetakis and P. Cook. This paper is linked [here](https://ieeexplore.ieee.org/document/1021072). The GTZAN dataset is the most used public dataset for machine learning for music genre classification. This dataset contains 1000 audio files, each 30 seconds in length. These files are divided into 10 different genres, each with 100 corresponding audio files. The dataset I downloaded includes these audio files as spectrograms, which were created using the librosa library. So, in total, my complete dataset includes 100 spectrograms for each of the 10 genres I will be able to classify: blues, classical, country, disco, hiphop, jazz (only has 99 files), metal, pop, reggae, and rock. This gives me a total of 999 spectrographs. Here are a few spectrograms from my dataset:

*Classification: Blues*

![Blues Spectrogram](/data_samples/blues00009.png "Classification: Blues")

*Classification: Hip Hop*

![Hip Hop Spectrogram](/data_samples/hiphop00003.png "Classification: Hip Hop")

*Classification: Metal*

![Metal Spectrogram](/data_samples/metal00002.png "Classification: Metal")


I compiled the images, which were originally split into 10 seperate folders for each genre, into one csv file containing the image name and the correct genre classification to use as input in my neural network for both training and validation. Then, I used scikit test_train_split to split my csv entries randomly into 70% training data and 30% validation data. The small Python code I used to do this is located in the [split_data.py] file in this GitHub.

For the "unknown" testing dataset, I ran 20 of my own song choices, 2 per genre as classified by Apple Music, through a [Spectrogram Creator](https://convert.ing-now.com/audio-spectrogram-creator/).


## Part 3
