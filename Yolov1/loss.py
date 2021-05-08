import torch
import torch.nn as nn
import torch.functional as F
from torch.autograd import Variable
import hyperparameter as hp

class loss(nn.Module):
    def __init__(self):
        self.S = 7
        self.B = 1
        self.l_coord = 5.0
        self.l_noobj = 0.5

    def compute_iou(self, box1, box2):
        N = box1.size(0)
        M = box2.size(0)
        lt = torch.max(
            box1[:, :2].unsqueeze(1).expand(N, M, 2),
            box2[:, :2].unsqueeze(0).expand(N, M ,2)
        )                           #lefttop
        rb = torch.min(
            box1[:, 2:].unsqueeze(1).expand(N, M, 2),
            box2[:, 2:].unsqueeze(0).expand(N, M, 2)
        )                           #rightbottom
        wh = rb-lt         #(N,M,2)
        wh[wh<0] = 0
        interarea = wh[:, :, 0]*wh[:, :, 1]     #(N,M)
        area1 = (box1[:, 2]-box1[:, 0])*(box1[:, 3]-box1[:, 1])
        area2 = (box2[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area1 = area1.unsqueeze(1).expand_as(interarea)
        area2 = area2.unsqueeze(0).expand_as(interarea)
        iou = interarea/(area1+area2-interarea)

        return iou

    def forward(self, pred, target):    #pred，target(batchsize, 7*7*30)
        containobj = target[:, :, :, 4]>0
        noobj = target[:, :, :, 4] == 0
        containobj = containobj.unsqueeze(-1).expand_as(target)
        noobj = noobj.unsqueeze(-1).expand_as(target)
        pred_containobj = pred[containobj].view(-1, 25) #select all grid in a batch that contains an obj
        target_containobj = target[containobj].view(-1, 25)
        box_pred = pred_containobj[:, :5]
        class_pred = pred_containobj[:, 5:]     #if obj, box and class
        box_target = target_containobj[:, :5]
        class_target = target_containobj[:, 5:]

        pred_noobj = pred[noobj].view(-1, 25) #select all grids no obj
        target_noobj = pred[noobj].view(-1, 25) #select all grids no obj in target
        noobj_mask = torch.ByteTensor(pred_noobj.size())
        noobj_mask.zero_()
        noobj_mask[:, 4] = 1
        confidence_pred = pred_noobj[noobj_mask]
        confidence_target = target_noobj[noobj_mask]
        noobj_loss = F.mse_loss(
            confidence_pred,
            confidence_target,
            size_average=False
        )

        containobj_res_obj = torch.ByteTensor(box_target.size())
        containobj_res_obj.zero_()

        box_iou = torch.zeros(box_target.size())
        for i in range(0, box_target.size()[0]):
            box1_predcontain_obj = box_pred[i].view(-1, 5)
            box1_xyxy = Variable(torch.FloatTensor(box1_predcontain_obj.size()))
            box1_xyxy[:, :2] = box1_predcontain_obj[:, :2]-0.5*self.S*box1_predcontain_obj[:, 2:4]
            box1_xyxy[:, 2:4] = box1_predcontain_obj[:, :2]+0.5*self.S*box1_predcontain_obj[:, 2:4]
            box2_targetcontain_obj = box_target[i].view(-1, 5)
            box2_xyxy = Variable(torch.FloatTensor(box2_targetcontain_obj.size()))
            box2_xyxy[:, :2] = box2_targetcontain_obj[:, :2]-0.5*self.S*box2_targetcontain_obj[:, 2:4]
            box2_xyxy[:, 2:4] = box2_targetcontain_obj[:, :2] + 0.5 * self.S * box2_targetcontain_obj[:, 2:4]
            iou = self.compute_iou(box1_xyxy[:, :4], box2_xyxy[:, :4])
            containobj_res_obj[i] = 1
            box_iou[i, 4] = iou

        res_predcontain = box_pred[containobj_res_obj].view(-1, 5)
        res_targetcontain = box_target[containobj_res_obj].view(-1, 5)
        res_iou = box_iou[containobj_res_obj].view(-1, 5)

        coordinate_loss = F.mse_loss(
            res_predcontain[:, :2],
            res_targetcontain[:, :2],
            size_average=False
        )+F.mse_loss(
            torch.sqrt(res_predcontain[:, 2:4]),
            torch.sqrt(res_targetcontain[:, 2:4]),
            size_average=False
        )

        res_loss = F.mse_loss(
            res_predcontain[:, 4],
            res_iou[:, 4],
            size_average=False
        )

        class_loss = F.mse_loss(
            class_pred,
            class_target,
            size_average=False
        )

        return (self.l_coord*coordinate_loss+class_loss+self.l_noobj*noobj_loss+res_loss)/hp.batchsize