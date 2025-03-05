import torch, torchvision
import copy

class Nop:
    def __init__(self, *args, **kwargs):
        pass
    def __call__(self, unsup_imgs):
        return 0

class FixMatch:
    # https://proceedings.neurips.cc/paper/2020/hash/06964dce9addb1c5cb5d6e3d9838f733-Abstract.html
    def __init__(self, model, weak_augment, strong_augment):
        self.model = model
        self.weak_augment = weak_augment
        self.strong_augment = strong_augment

    def __call__(self, epoch, imgs, confidence=0.95):
        weak_imgs = self.weak_augment(imgs)
        with torch.no_grad():
            weak_logits = self.model(weak_imgs)
        weak_probs = weak_logits.softmax(1)
        max_probs, weak_labels = weak_probs.max(1)
        ix = max_probs >= confidence
        strong_imgs = self.strong_augment(imgs[ix])
        return torch.nn.functional.cross_entropy(self.model(strong_imgs), weak_labels[ix]) if ix.sum() > 0 else 0

class MixMatch:
    # https://proceedings.neurips.cc/paper_files/paper/2019/hash/1cd138d0499a68f4bb72bee04bbec2d7-Abstract.html
    def __init__(self, model, weak_augment, strong_augment):
        self.model = model
        self.augment = weak_augment

    def __call__(self, epoch, imgs, K=2, T=0.5, alpha=0.75):
        with torch.no_grad():
            avg_probs = sum(self.model(self.augment(imgs)).softmax(1) for _ in range(K)) / K
        # sharpen
        labels = (avg_probs**(1/T)) / torch.sum(avg_probs**(1/T), 1, True)
        # mixup
        mixup = torchvision.transforms.v2.MixUp(num_classes=avg_probs.shape[1])
        imgs, labels = mixup(imgs, labels)
        # loss
        probs = self.model(imgs).softmax(1)
        return torch.mean((probs - labels)**2)

# TODO: implement ReMixMatch

class DINO:
    # https://arxiv.org/abs/2104.14294
    # version without multi-crop augmentation
    def __init__(self, model, weak_augment, strong_augment):
        self.student = model
        self.teacher = copy.deepcopy(model)
        self.augment = strong_augment
        self.C = 0

    def __call__(self, epoch, imgs, tps=0.1, m=0.9, l=0.99):
        tpt = min(epoch/30, 1)*(0.07-0.04) + 0.04
        def H(t, s):
            s = torch.nn.functional.log_softmax(s / tps, 1)
            t = torch.softmax((t-self.C)/tpt, 1)
            return -(t*s).sum(1).mean()
        imgs1, imgs2 = self.augment(imgs), self.augment(imgs)
        student_logits1, student_logits2 = self.student(imgs1), self.student(imgs2)
        with torch.no_grad():
            teacher_logits1, teacher_logits2 = self.teacher(imgs1), self.teacher(imgs2)
        loss = H(teacher_logits1, student_logits2)/2 + H(teacher_logits2, student_logits1)/2
        # update teacher
        for param_t, param_s in zip(self.teacher.parameters(), self.student.parameters()):
            param_t.data = l*param_t.data + (1-l)*param_s.data
        self.C = m*self.C + (1-m)*torch.cat((teacher_logits1, teacher_logits2)).mean(0, True)
        return loss
