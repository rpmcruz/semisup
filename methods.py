import torch
import albumentations as A

class UnsupervisedSplit:
    def __init__(self, ds, sup_percentage):
        self.ds = ds
        self.sup_ix = frozenset(int(i) for i in torch.randperm(len(ds))[:int(len(ds)*sup_percentage)])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return self.ds[i], i in self.sup_ix

def image_to_albumentation(d):
    d['image'] = d['image'].permute(1, 2, 0).cpu().numpy()
    return d

def apply_each(aug, d):
    import numpy as np
    # dataloaders converts data to dict of lists
    # albumentations only supports individual dictionaries
    # therefore we need to convert dict of lists => list of dicts, and then back
    device = d['image'].device
    l = [aug(**image_to_albumentation({key: d[key][i] for key in d})) for i in range(len(d['image']))]
    d = {key: torch.tensor(np.stack([el[key] for el in l]), device=device) for key in l[0]}
    d['image'] = d['image'].permute(0, 3, 1, 2)
    return d

def fixmatch(model, d, tau=0.95):
    hard_augment = A.augmentations.dropout.coarse_dropout.CoarseDropout()
    d2 = apply_each(hard_augment, d)
    key = [k for k in ['label', 'labels', 'masks'] if k in d][0]
    probs = model.forward_predict(d)[key]
    labels = d[key]
    probs, pseudo_labels = probs.max(1)
    ix = probs >= tau
    d = d2
    d[key] = pseudo_labels
    loss = model.forward_loss(d)
    return ix * loss
