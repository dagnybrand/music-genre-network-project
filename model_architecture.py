import torch
import torch.nn as nn
import torchvision 

class CNN(nn.Module):
    def __init__(self, numChannels = 3, numClasses = 10):
        super(CNN, self).__init__()
        self.classes = numClasses

        self.conv1 = nn.Conv2d(in_channels = numChannels, out_channels=64, kernel_size=(7, 7), stride = (6, 6))
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels=128, kernel_size=(3, 3), stride = (1, 1))
        self.conv4 = nn.Conv2d(in_channels = 128, out_channels=64, kernel_size=(3, 3), stride = (1, 1))

        self.relu = nn.ReLU()

        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = nn.Linear(in_features=10240, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=10)


    def evaluate(self, model, dataloader, classes):

            # We need to switch the model into the evaluation mode
            model.eval()

            # Prepare to count predictions for each class
            correct_pred = {classname: 0 for classname in classes}
            total_pred = {classname: 0 for classname in classes}

            # For all test data samples:
            for data in dataloader:
                images, labels = data
                outputs = model(images)
                _, predictions = torch.max(outputs, 1)

                # Count the correct predictions for each class
                for label, prediction in zip(labels, predictions):

                    # If you want to see real and predicted labels for all samples:
                    # print("Real class: " + classes[label] + ", predicted = " + classes[prediction])

                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1

            # Calculate the overall accuracy on the test set
            acc = sum(correct_pred.values()) / sum(total_pred.values())

            return acc


    def forward(self, x):
            #x = resize(x, size=[256])

            x = self.conv1(x)
            x = self.relu(x)

            x = self.conv2(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.conv4(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)

            return x