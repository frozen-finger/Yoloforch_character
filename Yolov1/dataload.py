import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import hyperparameter as hp
import matplotlib.pyplot as plt

def dataload(Dataset):
    def _init_(self, filedir, filename, transform):
        self.imsize = hp.width
        self.dir = filedir      #filedir where data in(txt), each line :image name, x1, x2, y1, y2, c
        self.transform = transform
        self.images = []
        self.boxes = []
        self.classes = []
        self.filename = filename

        with open(self.filename, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip().split()
            self.images.append(line[0])
            box = []
            for i in range(1, 5):
                box.append(float(line[i]))
            self.boxes.append(torch.tensor(box))
            self.classes.append(torch.LongTensor(int(line[5])))

    def _getitem_(self, id):
        imgname= self.images[id]
        img = plt.imread(os.path.join(self.dir, imgname))
        boxes = self.boxes[id].clone()
        label = self.classes[id].clone()
        h, w, _ = img.shape
        boxes /= torch.tensor([w, h, w, h]).expand_as(boxes)        #relative position
        target = self.encoder(self, boxes, label)           #generate7*7*25
        img = self.transform(img)
        return img, target

    def encoder(self, boxes, label):            #convert to a tensor 25 dimension, xcenter, ycenter, w, h, confidence, 20classes
            grid_num = hp.gridnum
            grid_size = 1.0/grid_num
            target = torch.zeros(grid_num, grid_num, 25)
            wh = boxes[2:]-boxes[:2]
            center = (boxes[2:]+boxes[:2])/2
            gr = (center*grid_num).floor()
            target[int(gr[1]), int(gr[0]), 4] = 1       #confidence 1
            target[int(gr[1]), int(gr[0]), int(label)+4] = 1 #class
            xy = gr*grid_size
            delta_center = (center-xy)/grid_size
            target[int(gr[1]), int(gr[0]), :2] = delta_center
            target[int(gr[1]), int(gr[0]), 2:4] = wh
            return target

    