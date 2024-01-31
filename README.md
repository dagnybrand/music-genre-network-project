# music-genre-network-project
Author: Dagny Brand

## Part 0
This repository holds the Semester Project for CSE 40868. For the project, I have chosen to do the Music Genre Classification pre-defined project topic.

This project will (hopefully, if all goes well) utilize an Anaconda virtual environment to run PyTorch.

## Part 1

This network is intended to classify the musical genre of input audio samples. For the brainstorming aspect of this project, I will assume my network will take in calculated audio values instead of a song itself. I started by doing research on how humans can tell the difference between different musical genres. (Acousterr)[https://www.acousterr.com/blog/how-to-identify-different-genres-of-music] provided me with lots of explanation on how to identify the difference. There are at least three major elements that I (or my classifier) should watch for.

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


 I need to learn how these features translate to recognizable data for a network to learn from. After exploring the (https://github.com/mdeff/fma) GitHub and dataset as well as consulting ChatGPT on which musical features are needed to classify genres, there are a few features I think are needed to extract in order to classify audio. These include **mel-frequency cepstral coefficients (MFCCs)** (coefficients that represent the power spectrum of a sound -- can help identify certain instruments), a **chromogram** (energy distribution of pitch classes -- can help identify instruments), **zero-crossing rate** (rate at which signal changes sign -- helps identify noisiness or percussiveness), **spectral features** (such as the centroid, contrast, roll-off, and bandwidth -- provide information about the distribution of energy in different frequency bands), **tonnetz** (temporal features -- capture overall trends), and the **root mean square** (captures energy or loudness). I can use the (librosa)[https://librosa.org/doc/latest/feature.html] library to extract these features from my own input songs, as was done in the reference GitHub to compile its dataset. 


 To create this classifier, I need three data subsets: one for training, one for validation, and one for final testing. To create the training and validation datasets, I can use the reference GitHub dataset along with my own dataset which I will create using librosa to make one large data set. I will then use scikit's test_train_split to split this dataset randomly into 
