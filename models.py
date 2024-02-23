# Torchvision models do not have a consistent API. Wrapper to make it so.
# Since torchvision object detection models compute the forward pass and the
# loss at the same time, we need to do so for the other models as well.

import torchvision
import torch

def dice_loss(logits, targets, smooth=1e-6):
    num_classes = logits.shape[1]
    probs = torch.softmax(logits, 1)
    targets = torch.nn.functional.one_hot(targets, num_classes)
    targets = targets.permute(0, 3, 1, 2)
    num = 2*torch.sum(targets*probs, [1, 2, 3]) + smooth
    den = torch.sum(targets + probs, [1, 2, 3]) + smooth
    dice = num / den
    return 1-dice

class Classification(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = torchvision.models.resnet50(num_classes=num_classes)

    def forward_loss(self, images, targets):
        preds = self.model(images)
        labels = torch.stack([t['label'] for t in targets])
        return torch.nn.functional.cross_entropy(preds, labels, reduction='none')

    def forward_predict(self, images):
        return {'label': torch.softmax(self.model(images), 1)}

class Segmentation(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = torchvision.models.segmentation.deeplabv3_resnet50(num_classes=num_classes)

    def forward_loss(self, images, targets):
        preds = self.model(images)['out']
        masks = torch.stack([t['masks'] for t in targets])
        return torch.nn.functional.cross_entropy(preds, masks, reduction='none') + \
            dice_loss(preds, masks)

    def forward_predict(self, images):
        return {'masks': torch.softmax(self.model(images)['out'], 1)}

class Detection(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = torchvision.models.detection.fcos_resnet50_fpn(num_classes=num_classes)

    def forward_loss(self, images, targets):
        losses = self.model(images, targets)['out']
        return sum(losses.values())

    def forward_predict(self, images):
        self.model.eval()
        preds = self.model(images)
        self.model.train()
        return preds
