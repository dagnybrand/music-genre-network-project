import torch
import torchvision
import torch.nn as nn
import model_architecture
from torchvision.transforms import ToTensor
from sklearn import preprocessing
import cv2
import numpy as np

def get_validation_sample(le, transform = ToTensor(), img='validation_sample/country00044.png', genre='Country'):
    image = cv2.imread(img)
    if transform:
        image = transform(image)
    label = le.transform([genre])

    return (image, label[0])

def test_model(test_sample, model_source = 'first_solution.pth'):
    model = model_architecture.CNN(numChannels = 3, numClasses = 10)
    model.load_state_dict(torch.load(model_source))
    test_sample = test_sample.reshape([1, 3, 288, 432])
    output = model(test_sample)
    _, pred = torch.max(output, 1)
    return pred

    
def __main__():
    le = preprocessing.LabelEncoder()
    le.fit(['Blues', 'Classical', 'Country', 'Disco', 'Hip Hop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock'])
    sample = get_validation_sample(le)
    pred = test_model(sample[0])
    pred = pred.numpy()[0]
    print(f"The song genre is {le.inverse_transform([pred])[0]}")

__main__()