import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import tqdm
import datetime
from pathlib import Path

from dataloader import brats3dDataset
from loss import DiceLoss, dice_coef_metric
from model import Model

NUM_EPOCHS = 50
BATCH_SIZE = 1
LR = 0.0001
CHANNELS = 40
SPLIT_FRAC = 0.20
FORCE_CPU = False

def gettime():
    now = datetime.datetime.now()
    return now.strftime('%H%M%S')

now = datetime.datetime.now()
start_time = now.strftime('%y%m%d_%H%M%S')
Path('models').mkdir(parents=True, exist_ok=True)
logPath = os.path.join('models', start_time + '.log')

def add2log(s,logPath=logPath,display=True):
    with open(logPath, 'a') as f:
        if display:
            print(s, flush=True, end='')
        f.write(s)
    return None

add2log(f"Channels is {CHANNELS}.\n")
add2log(f"NumEpochs is {NUM_EPOCHS}.\n")
add2log(f"Batch Length is {BATCH_SIZE}.\n")
add2log(f"Learning rate is {LR}.\n")
add2log(f"Valid/Train ratio is {SPLIT_FRAC}.\n")

dataPath = "/projects/ab57/data/BraTS2021_Training_Combined/"
fullDataset = brats3dDataset(dataPath)
add2log(f"\tThere are {len(fullDataset)} images in total.\n")

# Split into training and validation
valid_size = int(SPLIT_FRAC * len(fullDataset))
train_size = len(fullDataset) - valid_size
train_dataset, valid_dataset = random_split(fullDataset, [train_size, valid_size])
add2log(f"\tThere are {len(train_dataset)} training images, and {len(valid_dataset)} validation images.\n")

# Load data
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
add2log(f'{gettime()} Data is loaded.\n')

# Set device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

if FORCE_CPU:
    device = torch.device('cpu')
    print(f"\nDevice forced to cpu")
else:
    print(f"\nDevice used is {device}.")

# Initialize model
model = Model(in_channels=4, n_classes=3, n_channels=CHANNELS)
model.to(device)
add2log(f"\n{gettime()} Model was initialized.\n")

# Set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
add2log(f"\tOptimizer Adam with learning rate {LR}.\n")

# loss and score
criterion = DiceLoss()
score_fun = dice_coef_metric

# Training
add2log("\n--- Starting Training ---\n")
ID = gettime()
add2log(f"\nSaving losses and scores as {ID}\n")

epochLosses = []
epochScores = []
epochTrainScores = []

best_score = [0.0, -1]

try:
    for epoch in range(NUM_EPOCHS):
        add2log(f'\n-------Epoch {epoch+1}-------\n')
        add2log(f"{gettime()} started epoch {epoch+1}.\n")

        # Training Loop
        add2log('\n\t---Training-------------\n\n')
        model.train()
        trainlosses = []
        trainscores = []
        batchloop = tqdm.tqdm(train_dataloader)
        for x,y in batchloop:
            optimizer.zero_grad()
            x = x.to(device)  # Use cuda if available
            y = y.to(device)
            
            y_pred = model(x)           # Forward pass
            loss = criterion(y_pred,y)  # Compute loss
            loss.backward()             # Backward pass
            optimizer.step()

            batchloop.set_description(f"Epoch number {epoch+1}, Loss: {loss.item()}")
            trainlosses.append(loss.item())

            score = score_fun(y_pred,y)
            trainscores.append(score)

        avgloss = np.asarray(trainlosses).mean()
        add2log(f"\nThe average training loss was {avgloss}.\n")
        epochLosses.append(avgloss)

        avgTrainScore = np.asarray(trainscores).mean()
        add2log(f"The average training score was {avgTrainScore}.\n")
        epochTrainScores.append(avgTrainScore)

        # Validation loop
        add2log('\n\t---Testing-------------\n\n')
        model.eval()
        with torch.no_grad():
            scores = []
            validloop = tqdm.tqdm(valid_dataloader)
            for x,y in validloop:
                x = x.to(device)
                y = y.to(device)
                y_pred = model(x)

                score = score_fun(y_pred,y)
                scores.append(score)

            avgscore = np.asarray(scores).mean()
            epochScores.append(avgscore)
            add2log(f"\nThe average testing score was {avgscore}.\n")

            # Save best model
            if avgscore > best_score[0]:
                add2log('The score improved!\n')
                best_score[0] = avgscore
                best_score[1] = epoch+1
                rootpath = './'
                modelPath = os.path.join(rootpath, 'models', start_time + '_' + '.pt')
                add2log(f"Saving model from epoch {epoch+1}.\n{modelPath}\n")
                torch.save(model.state_dict(), modelPath)  
        
        with open(os.path.join('models',f"{ID}-testing-scores.txt"), "a") as myfile:
            myfile.write(str(avgscore) + '\n')
        with open(os.path.join('models',f"{ID}-training-scores.txt"), "a") as myfile:
            myfile.write(str(avgTrainScore) + '\n')
        with open(os.path.join('models',f"{ID}-training-losses.txt"), "a") as myfile:
            myfile.write(str(avgloss) + '\n')

    add2log(f"\n{gettime()} is done training.\n")

except KeyboardInterrupt:
    add2log(f'\n{gettime()} Training was interrupted by KeyboardInterrupt.\n')

add2log(f"The last training score was {epochTrainScores[-1]}.\n")
add2log(f"The best score achieved during epoch {best_score[1]}, with an average validation score {best_score[0]}.")