import datetime
import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torchvision.models import vgg16, VGG16_Weights
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

training_data, test_data = random_split(data, [math.ceil(len(data) * 0.8), math.floor(len(data) * 0.2)])

training_loader = torch.utils.data.DataLoader(training_data, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

class Model(nn.Module):
    def __init__(self, num_classes=16):
        super(Model, self).__init__()
        
        self.efficientnet = vgg16()
        self.efficientnet.classifier = nn.Sequential(
            nn.Linear(25088, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, images):
        return self.efficientnet(images)

model = Model().to(device)
# print(model)
weights = VGG16_Weights.DEFAULT
loss_func = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.004, momentum=0.88)

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(tqdm(training_loader)):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        preprocess = weights.transforms()
        batch = preprocess(images.to(device)).unsqueeze(0)[0]
        outputs = model(batch)
        loss = loss_func(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


writer = SummaryWriter('runs/fashion_trainer')
epoch_number = 0

EPOCHS = 50

best_vloss = 1_000_000.

train_loss = []
val_loss = []
acuuracys = []

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    running_vloss = 0.0

    model.eval()
    correct = 0
    with torch.no_grad():
        for i, vdata in enumerate(tqdm(test_loader)):
            images, labels = vdata
            images = images.to(device)
            labels = labels.to(device).float()
            preprocess = weights.transforms()
            batch = preprocess(images).unsqueeze(0)[0]
            voutputs = model(batch)
            label = torch.argsort(voutputs)
            # print(labels)
            # print(label)
            vloss = loss_func(voutputs.float(), labels)
            running_vloss += vloss.item()

            for j in range(len(labels)):
                if(labels[j][label[j][-1]] == 1 or labels[j][label[j][-2]] == 1 or labels[j][label[j][-3]] == 1 or labels[j][label[j][-4]] == 1):
                    correct += 1

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss), '\nAccuracy: ', correct/(len(test_loader)*32))
    acuuracys.append(correct/(len(test_loader)*32))
    train_loss.append(avg_loss)
    val_loss.append(avg_vloss)

    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}'.format(epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1

torch.save(model.state_dict(), "DeepLearning/models/vgg_50.pt")

print(train_loss)
print(val_loss)

plt.title("VGG16 Loss")
plt.xlabel("Epochs")
plt.ylabel("Average Loss")
plt.legend(['train loss', 'val loss']) 
plt.plot(train_loss)
plt.plot(val_loss)
plt.show()

plt.plot(acuuracys)
plt.show()