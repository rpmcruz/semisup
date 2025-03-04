import torch, torchvision
import copy

class Nop:
    def __init__(self, *args, **kwargs):
        pass
    def __call__(self, unsup_imgs):
        return 0

class FixMatch:
    # https://proceedings.neurips.cc/paper/2020/hash/06964dce9addb1c5cb5d6e3d9838f733-Abstract.html
    def __init__(self, model, weak_transform, strong_transform):
        self.model = model
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __call__(self, imgs, confidence=0.95):
        weak_imgs = self.weak_transform(imgs)
        with torch.no_grad():
            weak_logits = self.model(weak_imgs)
        # fixmatch loss (unsupervised)
        weak_probs = weak_logits.softmax(1)
        max_probs, weak_labels = weak_probs.max(1)
        ix = max_probs >= confidence
        strong_imgs = self.strong_transform(imgs[ix])
        return torch.nn.functional.cross_entropy(self.model(strong_imgs), weak_labels[ix]) if ix.sum() > 0 else 0

class MixMatch:
    # https://proceedings.neurips.cc/paper_files/paper/2019/hash/1cd138d0499a68f4bb72bee04bbec2d7-Abstract.html
    def __init__(self, model, weak_transform, strong_transform):
        self.model = model
        self.weak_transform = weak_transform

    def __call__(self, imgs, K=2, T=0.5, alpha=0.75):
        with torch.no_grad():
            avg_probs = sum(self.model(self.weak_transform(imgs)).softmax(1) for _ in range(K)) / K
        # sharpen
        labels = (avg_probs**(1/T)) / torch.sum(avg_probs**(1/T), 1, True)
        # mixup
        mixup = torchvision.transforms.v2.MixUp(num_classes=avg_probs.shape[1])
        imgs, labels = mixup(self.weak_transform(imgs), labels)
        # loss
        probs = self.model(imgs).softmax(1)
        return torch.mean((probs - labels)**2)

# FIXME: DINO results are bad; prolly not well implemented
class DINO:
    # https://arxiv.org/abs/2104.14294
    def __init__(self, model, weak_transform, strong_transform):
        self.student = model
        self.teacher = copy.deepcopy(model)
        self.weak_transform = weak_transform

    def __call__(self, imgs, teacher_momentum=0.999, student_temp=0.1, teacher_temp=0.04):
        imgs = self.weak_transform(imgs)
        # update teacher
        for param_t, param_s in zip(self.teacher.parameters(), self.student.parameters()):
            param_t.data = teacher_momentum*param_t.data + (1-teacher_momentum)*param_s.data
        # teach student
        with torch.no_grad():
            teacher_logits = self.teacher(imgs)
        teacher_probs = torch.softmax((teacher_logits - teacher_logits.mean(0, True)) / teacher_temp, 1)
        student_logits = self.student(imgs)
        student_logprobs = torch.nn.functional.log_softmax(student_logits / student_temp, 1)
        # loss
        return -torch.mean(torch.sum(teacher_probs * student_logprobs, 1))
