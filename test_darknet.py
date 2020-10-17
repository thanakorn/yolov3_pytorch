import cv2 as cv
import numpy as np
import torch

from models.darknet import parse_cfg, create_modules, Darknet
from torch.autograd import Variable

def get_test_input():
    img = cv.imread("dog-cycle-car.png")
    img = cv.resize(img, (416,416))          #Resize to the input dimension
    img_ = img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W 
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_

if __name__ == '__main__':
    model = Darknet('cfg/yolov3.cfg')
    # print(model)
    test_input = get_test_input()
    detection = model(test_input)
    print(detection.size())
    print(detection)