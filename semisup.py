import torch, torchvision
import copy

class Supervised:
    def __init__(self, model, weak_transform, strong_transform):
        self.model = model
        self.transform = weak_transform

    def compute_loss(self, sup_imgs, sup_labels, unsup_imgs):
        sup_imgs = self.transform(sup_imgs)
        logits = self.model(sup_imgs)
        supervised_loss = torch.nn.functional.cross_entropy(logits, sup_labels)
        return supervised_loss, 0

class FixMatch:
    def __init__(self, model, weak_transform, strong_transform):
        self.model = model
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def compute_loss(self, sup_imgs, sup_labels, unsup_imgs, confidence=0.95):
        # batch normalization is affected by doing two fpasses
        # therefore make a single fpass
        sup_imgs = self.weak_transform(sup_imgs)
        weak_unsup_imgs = self.weak_transform(unsup_imgs)
        strong_unsup_imgs = self.strong_transform(unsup_imgs)
        logits = self.model(torch.cat((sup_imgs, weak_unsup_imgs, strong_unsup_imgs)))
        sup_logits, weak_unsup_logits, strong_unsup_logits = torch.split(logits, (len(sup_imgs), len(weak_unsup_imgs), len(strong_unsup_imgs)))
        # supervised loss
        supervised_loss = torch.nn.functional.cross_entropy(sup_logits, sup_labels)
        # fixmatch loss (unsupervised)
        weak_unsup_probs = weak_unsup_logits.detach().softmax(1)
        max_probs, weak_unsup_labels = weak_unsup_probs.max(1)
        ix = max_probs >= confidence
        unsupervised_loss = torch.nn.functional.cross_entropy(strong_unsup_logits[ix], weak_unsup_labels[ix]) if ix.sum() > 0 else 0
        return supervised_loss, unsupervised_loss

class MixMatch:
    # https://proceedings.neurips.cc/paper_files/paper/2019/hash/1cd138d0499a68f4bb72bee04bbec2d7-Abstract.html
    def __init__(self, model, weak_transform, strong_transform):
        self.model = model
        self.weak_transform = weak_transform

    def __call__(self, model, imgs, K=2, T=0.5, alpha=0.75):
        with torch.no_grad():
            avg_probs = sum(torch.softmax(model(self.weak_augment(imgs)), 1) for _ in range(K)) / K
        # sharpen
        labels = (avg_probs**(1/T)) / torch.sum(avg_probs**(1/T))
        # mixup
        mixup = torchvision.transforms.v2.MixUp(num_classes=avg_probs.shape[1])
        imgs, labels = mixup(imgs, labels)
        # loss
        probs = torch.softmax(model(imgs), 1)
        return torch.mean((probs - labels)**2)

class DINO:
    # https://arxiv.org/abs/2104.14294
    def __init__(self, model, weak_transform, strong_transform):
        self.student = model
        self.teacher = copy.deepcopy(model)

    def __call__(self, imgs, teacher_momentum=0.999, student_temp=0.1, teacher_temp=0.04):
        # update teacher
        for param_t, param_s in zip(self.teacher.parameters(), self.student.parameters()):
            param_t.data = teacher_momentum*param_t.data + (1-teacher_momentum)*param_s.data
        # teach student
        with torch.no_grad():
            teacher_logits = self.teacher(imgs)
        teacher_probs = torch.softmax((teacher_logits - teacher_logits.mean(0)) / teacher_temp, 1)
        student_logits = self.student(imgs)
        student_probs = torch.softmax(student_logits / student_temp, 1)
        return -torch.mean(torch.sum(teacher_probs * torch.log(student_probs+1e-9), 1))