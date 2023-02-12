from subprocess import check_output
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import cv2
import glob
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# declare constants 
FILENAME = "test"
PREDICT_PATH = "d:\HACKATHONPROJECT\output\cnnimages\\"
TRAINED_MODEL_SAVE_PATH = "d:\HACKATHONPROJECT\model\pose_predict.pth"
TRAINING_IMAGE_PATH = "d:\HACKATHONPROJECT\data\\training_images\\"
IMAGE_PATH = "d:\HACKATHONPROJECT\output\images\\" + FILENAME + "\\"
JSON_PATH = "d:\HACKATHONPROJECT\output\json\\" + FILENAME + "\\"
CNN_IMAGE_PATH = "d:\HACKATHONPROJECT\output\cnnimages\\" + FILENAME + "\\"
COMMAND = "d:/HACKATHONPROJECT/openpose/bin/OpenPoseDemo.exe --video d:/HACKATHONPROJECT/data/" + FILENAME + ".mp4 --number_people_max 1 --net_resolution \"400x400\" --output_resolution \"1024x1024\" --disable_blending --write_images d:/HACKATHONPROJECT/output/images/" + FILENAME + " --write_json d:/HACKATHONPROJECT/output/json/" + FILENAME
FRAME_COUNT_FOR_TWIST = 2
FRAME_COUNT_FOR_DISPLAY = 30

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def main():
    # set the seed
    torch.manual_seed(0)

    # run openpose and get the data, 
    check_output(COMMAND, shell=True).decode()

    # get a list of images and jsons created by openpose
    image_list = os.listdir(IMAGE_PATH)
    json_list = os.listdir(JSON_PATH)

    # go through each and mark the images and json as front or back
    # count how many twirls and at what image
    change_count, change_images, pose_images = twirl_count(image_list, json_list)

    # save the pose images in a seperate folder
    print("Image Processing")
    #save_pose_images (pose_images)
    # save all the images in a different folder
    #save_all_images (image_list)

    # run those images through the ai model
    # mark the predicted pose in the images
    # prediction
    predictions = []
    output = prediction()
    for entry in output:
        entry = torch.rand(4)
        predictions.append(torch.argmin(entry))
    
    # train model
    #train_model()

    # make a video file from the image
    write_text_to_image_video(image_list, change_images, pose_images, predictions)

def prediction():
    print("Prediction start:")
    predictions = []
    transform = transforms.Compose([
                                transforms.RandomInvert(p=1),
                                transforms.Resize(32),
                                transforms.CenterCrop((32,32)),
                                transforms.ToTensor()])
    # load data
    dataset = datasets.ImageFolder(PREDICT_PATH, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    # load the model
    net = Net()
    net.load_state_dict(torch.load(TRAINED_MODEL_SAVE_PATH))
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        outputs = net(inputs)
        predictions.append(outputs)
    return predictions

def train_model():
    #transform data
    transform = transforms.Compose([
                                transforms.RandomInvert(p=1),
                                transforms.Resize(32),
                                transforms.CenterCrop((32,32)),
                                transforms.ToTensor()])
    #load data
    dataset = datasets.ImageFolder(TRAINING_IMAGE_PATH, transform=transform)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    
    #train model
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0)
    for epoch in range(100):  # loop over the dataset multiple times
        print("Number of epochs: " + str(epoch))
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 5 == 4:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20:.3f}')
                running_loss = 0.0
    print('Finished Training')
    #save model
    torch.save(net.state_dict(), TRAINED_MODEL_SAVE_PATH)
    

def write_text_to_image_video (image_list, change_images, pose_images, predictions):
    count = 0
    display_string_array = ["spread eagle", "toe jump", "salchow", "camel position", "flip", ""]
    display_string_index = 5
    display_count = 0
    prediction_index = 0
    out = cv2.VideoWriter("D:\\test.avi", cv2.VideoWriter_fourcc(*'DIVX'), 25, (1024, 1024))
    for index in range(len(image_list)):
        #print("Writing video:" + str((index / len(image_list)) * 100))
        if image_list[index] in change_images:
            count += 1
        if image_list[index] in pose_images:
            if display_count < FRAME_COUNT_FOR_DISPLAY:
                display_string_index = predictions[prediction_index]
            prediction_index += 1
            display_count = 0
        image = cv2.imread(IMAGE_PATH + image_list[index])
        font = cv2.FONT_HERSHEY_SIMPLEX
        org1 = (50, 50)
        org2 = (50, 100)
        fontScale = 1
        color = (255, 255, 255)
        thickness = 2
        image = cv2.putText(image, "twirl count: " + str(count), org1, font, fontScale, color, thickness, cv2.LINE_AA)
        if display_count < FRAME_COUNT_FOR_DISPLAY:
            image = cv2.putText(image, "pose: " + display_string_array[display_string_index], org2, font, fontScale, color, thickness, cv2.LINE_AA)
        #cv2.imwrite(ALTERED_IMAGE_PATH + image_list[index], image)
        out.write(image)
        display_count += 1
    out.release()
    return

def save_pose_images (images):
    for entry in images:
        check_output("copy " + IMAGE_PATH + entry + " " + CNN_IMAGE_PATH, shell=True).decode()
    return

def twirl_count (image_list, json_list):
    change_images = []
    pose_images = []
    change_count = []
    facing_direction = False
    prev_direction = False
    opposite_direction_count = 0
    count = 0
    # go through the json files and get the left and right shoulder positions
    # check if front or back 
    for index in range(len(image_list)):
        file = open(JSON_PATH  + json_list[index], "r")
        data = json.load(file)
        if data['people']:
            left = (data['people'][0]["pose_keypoints_2d"][2])
            right = (data['people'][0]["pose_keypoints_2d"][5])
        if left > right:
            facing_direction = 1
        else:
            facing_direction = 0
        if (facing_direction != prev_direction):
            opposite_direction_count += 1
        else:
            opposite_direction_count = 0
        if opposite_direction_count > FRAME_COUNT_FOR_TWIST:
            prev_direction = not prev_direction
            opposite_direction_count = 0
            if(prev_direction):
                count += 1
                change_images.append(image_list[index])
                change_count.append(count)
                if index + 15 < len(image_list):
                    pose_images.append(image_list[index + 15])
    #print(change_count, end="")
    return change_count, change_images, pose_images

if __name__ == "__main__":
    main()
