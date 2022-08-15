#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import argparse
import time
import os
import sys
import logging

from PIL import ImageFile

from smdebug import modes
from smdebug.profiler.utils import str2bool
from smdebug.pytorch import get_hook
import smdebug.pytorch as smd


#TODO: Import dependencies for Debugging andd Profiling


ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def test(model, test_loader, criterion, device, hook):
    '''
    EDIT: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    hook.set_mode(smd.modes.EVAL)
    acc, los = 0, 0
    with torch.no_grad():
        for inputs, datalabels in test_loader:
            inputs, datalabels = inputs.to(device), datalabels.to(device)
            outputs=model(inputs)
            _, preds = torch.max(outputs, 1)
            acc += torch.sum(preds == datalabels.data).item()
            loss=criterion(outputs, datalabels)
            los += loss.item() * inputs.size(0) 
    total_acc = acc // len(test_loader)
    total_loss = los // len(test_loader) 
    print('Test Accuracy:{}% \t Test Loss: {}'.format(100*total_acc,total_loss))
    return total_acc

def train(model, train_loader, validation_loader, criterion, optimizer, epochs, device, hook):
    '''
    EDIT: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    best_loss=1e6
    image_dataset={'train':train_loader, 'valid':validation_loader}
    loss_counter=0
    hook.set_mode(smd.modes.TRAIN)
    
    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            if phase=='train':
                model.train()
                hook.set_mode(smd.modes.TRAIN)
            los = 0.0
            acc = 0

            for inputs, datalabels in image_dataset[phase]:
                inputs=inputs.to(device)
                datalabels=datalabels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, datalabels)
                if phase=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                _, predictions = torch.max(outputs, 1)
                los += loss.item() * inputs.size(0)
                acc += torch.sum(predictions == datalabels.data)
            epoch_loss = los // len(image_dataset[phase])
            epoch_acc = acc // len(image_dataset[phase])
            if phase=='valid':
                if epoch_loss<best_loss:
                    best_loss=epoch_loss
                else:
                    loss_counter+=1
            logger.info("Epoch {},\t {} loss: {}, acc: {}, best loss: {}\n".format(epoch, phase, epoch_loss, epoch_acc, best_loss))
        if loss_counter==1:
            break            
    return model
        
    
def net():
    '''
    EDIT: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False   
    num_features=model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 512),
                             nn.ReLU(inplace=True), 
                             nn.Linear(512,133))
    return model

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    train_dir = os.path.join(data, 'train')
    test_dir = os.path.join(data, 'test')
    val_dir =os.path.join(data, 'valid')

    training_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()])
    valid_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()])
    testing_transform = transforms.Compose([
        transforms.RandomResizedCrop((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])
  
    train_set = torchvision.datasets.ImageFolder(root=train_dir, transform=training_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_set = torchvision.datasets.ImageFolder(root=test_dir, transform=testing_transform)
    test_loader  = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
    validation_set = torchvision.datasets.ImageFolder(root=val_dir, transform=valid_transform)
    validation_loader  = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True) 
    
    return train_loader, test_loader, validation_loader

def main(args):
    '''
    EDIT: Initialize a model by calling the net function
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on Device {device}")
    model=net()
    model=model.to(device)
    
    hook = smd.Hook.create_from_json_file()
    hook.register_module(model)

    '''
    EDIT: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    hook.register_loss(loss_criterion)
    
    '''
    EDIT: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    logger.info(int(args.batch_size))
    train_loader, test_loader, validation_loader = create_data_loaders(args.data, args.batch_size)
    model=train(model, train_loader, validation_loader, loss_criterion, optimizer, args.epochs, device,hook)
    
    '''
    EDIT: Test the model to see its accuracy
    '''
    logger.info("Test the Model")
    test(model, test_loader, loss_criterion, device,hook)
    
    '''
    EDIT: Save the trained model
    '''
    logger.info("Save the  Model")
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    EDIT: Specify any training args that you might need
    '''
    parser.add_argument("--batch_size", type=int, default=32, metavar="N", help="input batch size for training" )
    parser.add_argument("--epochs", type=int, default=5, metavar="N", help="number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.1, metavar="LR", help="learning rate")
    
    parser.add_argument("--model-dir", type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--data', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args=parser.parse_args()
    
    main(args)
