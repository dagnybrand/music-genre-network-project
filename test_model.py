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
    test_set_labels = [('at_last_(blues).png', 'Blues'), ('back_in_black_(rock).png', 'Rock'), 
                       ('california_(rock).png', 'Rock'), ('copperhead_road_(country).png', 'Country'), 
                       ('cruise_(country).png', 'Country'), ('cry_me_a_river_(classical).png', 'Classical'), 
                       ('dream_a_little_dream_of_me_(jazz).png', 'Jazz'), ('drinking_problem_(reggae).png', 'Reggae'),
                       ('enter_sandman_(metal).png', 'Metal'), ('good_life_(pop).png', 'Pop'),
                       ('heart_of_glass_(disco).png', 'Disco'), ('hedwig_theme_(classical).png', 'Classical'),
                       ('kung_fu_fighting_(disco).png', 'Disco'), ('maps_(pop).png', 'Pop'),
                       ('oh_well_(blues).png', 'Blues'), ('the_devil_in_i_(metal).png', 'Metal'),
                       ('this_is_the_life_(reggae).png', 'Reggae'), ('thrift_shop_(hiphop).png', 'Hip Hop'),
                       ('u_cant_touch_this_(hiphop).png', 'Hip Hop'), ('we_are_siamese_(jazz).png', 'Jazz')]
    
    samples = []
    for label in test_set_labels:
        samples.append(get_validation_sample(le, img=f'test_samples/{label[0]}', genre=label[1]))
    #sample = get_validation_sample(le, img='test_samples/back_in_black.png')
    #pred = test_model(sample[0])
    #pred = pred.numpy()[0]
    #print(f"The song genre is {le.inverse_transform([pred])[0]}")

    num_correct = 0
    for sample in samples:
        pred = test_model(sample[0])
        pred = pred.numpy()[0]
        if pred == sample[1]:
            num_correct += 1
        print(f"{sample[1]} \ {pred} -- The song genre is {le.inverse_transform([pred])[0]}")
    print(f"The test set accuracy is {num_correct/len(samples) * 100}%")

__main__()