import datetime
import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from torchvision.models import efficientnet_b2, densenet169, alexnet, vgg16
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print("Device: ", device)

class DataPreprocessing(Dataset):

    def __init__(self, max_samples=None):
        self.image_names = []
        self.labels = []
        #The original code opened the labels file from the zip file. I upload the labels file manually to the directory and open it manually here
        f = open('final_dataset1.csv','r')
        
        title = True
        for line in f:
            if (title):
                title = False
                continue
            items = str(line).split(",")

            
            label = items[2:]

            for i in range(len(label)):
                label[i] = float(label[i][0])
            if(len(label) == 15):
                label.append(0)
            label = torch.FloatTensor(label)


            if(label[10] == 1):
                continue

            self.labels.append(label)
            image_name = items[1]
            self.image_names.append(image_name)

            if max_samples and len(self.image_names) >= max_samples:
                break   

    def __getitem__(self, index):
        image_path = self.image_names[index]
        image = Image.open("datasets/images/"+image_path,'r').convert('RGB')
        image = image.resize((299,299))
        preprocess = transforms.Compose([ transforms.PILToTensor() ]) 
        image = preprocess(image)
        return image, self.labels[index]

    def __len__(self):
        return len(self.image_names)

data = DataPreprocessing()

training_data, test_data = random_split(data, [math.ceil(len(data) * 0.9995), math.floor(len(data) * 0.0005)])
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

print(len(test_data))

class EffNet(nn.Module):
    def __init__(self, num_classes=16):
        super(EffNet, self).__init__()
        
        self.efficientnet = vgg16()
        self.efficientnet.classifier = nn.Sequential(
            nn.Linear(25088, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, images):
        return self.efficientnet(images)

model = EffNet().to(device)
model.load_state_dict(torch.load('model_0'))

correct = 0

from sklearn.metrics import precision_recall_fscore_support
import numpy as np

def evaluate_model(model, images, labels):
    # Predict on the test set
    y_pred = model(images.float())
    # Convert predicted probabilities to binary predictions
    y_pred_binary = np.argsort(y_pred.cpu().detach().numpy())

    pred = [-1]*len(y_pred_binary)
    true = [-1]*len(y_pred_binary)
    for i in range(len(y_pred_binary)):
        for j in range(4):
            if(labels[i][-y_pred_binary[i][-j]] == 1.0):
                pred[i] = y_pred_binary[i][-j]
                true[i] = y_pred_binary[i][-j]
        if pred[i] == -1:
            pred[i] = y_pred_binary[i][-1]
            true[i] = np.argmax(labels[i].cpu()).item()
    
    # Compute precision, recall, and F1 score
    precision, recall, f1_score, _ = precision_recall_fscore_support(true, pred,average='weighted')
    
    return precision, recall, f1_score

precision, recall, f1_score = 0,0,0

for i, data in enumerate(tqdm(test_loader)):
    images, labels = data
    images = images.to(device)
    labels = labels.to(device)

    # # show image
    # image = F.to_pil_image(images[0])
    # image.show()

    # predict label
    outputs = model(images.float())
    print(labels)
    label = torch.argsort(outputs)
    print(outputs)
    # print(label)
    # print()

    # if label == 0:
    #     print('Atelectasis')
    # elif label == 1:
    #     print('Cardiomegaly')
    # elif label == 2:
    #     print('Consolidation')
    # elif label == 3:
    #     print('Edema')
    # elif label == 4:
    #     print('Effusion')
    # elif label == 5:
    #     print('Emphysema')
    # elif label == 6:
    #     print('Fibrosis')
    # elif label == 7:
    #     print('Hernia')
    # elif label == 8:
    #     print('Infiltration')
    # elif label == 9:
    #     print('Mass')
    # elif label == 10:
    #     print('No Finding')
    # elif label == 11:
    #     print('Nodule')
    # elif label == 12:
    #     print('Pleural_Thickening')
    # elif label == 13:
    #     print('Pneumonia')
    # elif label == 14:
    #     print('Pneumothorax')
    # elif label == 15:
    #     print('Covid-19')

    for j in range(len(labels)):
        if(labels[j][label[j][-1]] == 1 or labels[j][label[j][-2]] == 1 or labels[j][label[j][-3]] == 1 or labels[j][label[j][-4]] == 1):
            correct += 1
    precision, recall, f1_score = evaluate_model(model, images, labels)
    # else:
        # print("false")
    # print("--------------------------------------------------------------------")

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
print(correct/(len(test_data)))