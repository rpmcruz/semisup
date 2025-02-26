from torchvision.transforms import v2
import torch
import copy

def augmentations(do_hflip):
    soft_augment = v2.Compose((
        v2.RandomHorizontalFlip(p=0.5 if do_hflip else 0),
        v2.RandomAffine(0, (0.125, 0.125)),
    ))
    hard_augment = v2.RandAugment()
    return soft_augment, hard_augment

class Dummy:
    def __init__(self, **kwargs):
        pass

    def __call__(self, model, imgs):
        return torch.zeros((), requires_grad=True)

class FixMatch:
    # https://dl.acm.org/doi/abs/10.5555/3495724.3495775
    def __init__(self, soft_augment, hard_augment, **kwargs):
        self.soft_augment = soft_augment
        self.hard_augment = hard_augment

    def __call__(self, model, imgs, tau=0.95):
        soft_imgs = self.soft_augment(imgs)
        with torch.no_grad():
            soft_probs = torch.softmax(model(soft_imgs), 1)
        soft_probs_max, soft_labels = torch.max(soft_probs, 1)
        ix = soft_probs_max >= tau
        if ix.sum() == 0:  # ignore if nothing
            return torch.zeros((), requires_grad=True)
        hard_imgs = self.hard_augment(imgs[ix])
        logits = model(hard_imgs)
        return torch.nn.functional.cross_entropy(logits, soft_labels[ix])

class MixMatch:
    # https://proceedings.neurips.cc/paper_files/paper/2019/hash/1cd138d0499a68f4bb72bee04bbec2d7-Abstract.html
    def __init__(self, soft_augment, **kwargs):
        self.soft_augment = soft_augment

    def __call__(self, model, imgs, K=2, T=0.5, alpha=0.75):
        with torch.no_grad():
            avg_probs = sum(torch.softmax(model(self.soft_augment(imgs)), 1) for _ in range(K)) / K
        # sharpen
        labels = (avg_probs**(1/T)) / torch.sum(avg_probs**(1/T))
        # mixup
        lmbda = torch.distributions.Beta(alpha, alpha).sample([len(imgs)]).to(imgs.device)
        lmbda = torch.maximum(lmbda, 1-lmbda)
        ix = torch.randperm(len(imgs)).to(imgs.device)
        imgs = lmbda[:, None, None, None]*imgs + (1-lmbda[:, None, None, None])*imgs[ix]
        labels = lmbda[:, None]*labels + (1-lmbda[:, None])*labels[ix]
        # loss
        probs = torch.softmax(model(imgs), 1)
        return torch.mean((probs - labels)**2)

class DINO:
    # https://arxiv.org/abs/2104.14294
    def __init__(self, teacher, **kwargs):
        self.teacher = copy.deepcopy(teacher)

    def __call__(self, student, imgs, teacher_momentum=0.999, student_temp=0.1, teacher_temp=0.04):
        # update teacher
        for param_t, param_s in zip(self.teacher.parameters(), student.parameters()):
            param_t.data = teacher_momentum*param_t.data + (1-teacher_momentum)*param_s.data
        # teach student
        with torch.no_grad():
            teacher_logits = self.teacher(imgs)
        teacher_probs = torch.softmax((teacher_logits - teacher_logits.mean(0)) / teacher_temp, 1)
        student_logits = student(imgs)
        student_probs = torch.softmax(student_logits / student_temp, 1)
        return -torch.mean(torch.sum(teacher_probs * torch.log(student_probs+1e-9), 1))
