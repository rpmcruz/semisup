import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('method')
parser.add_argument('--supervised', default=0.5, type=float)
parser.add_argument('--lamda', default=1, type=float)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batchsize', default=32, type=int)
args = parser.parse_args()

import torch, torchvision
from torchvision.transforms import v2
from time import time
import data, models, methods

device = 'cuda' if torch.cuda.is_available() else 'cpu'

################################## DATA ##################################

ds = getattr(data, args.dataset)

bbox_params = A.BboxParams('pascal_voc') if ds.task == 'detection' else None
transform = A.Compose([
    A.Resize(int(ds.imgsize*1.05), int(ds.imgsize*1.05)),
    A.RandomCrop(ds.imgsize, ds.imgsize),
    A.HorizontalFlip(),
    A.RandomBrightnessContrast(p=1),
    A.Normalize(0, 1),
    ToTensorV2(),
], bbox_params)

ds = ds('/data/toys', True, transform)
tr = ds
tr = methods.UnsupervisedSplit(tr, args.supervised)

################################## BATCHES ##################################

def collate_fn(batch):
    types = {'label': torch.int64, 'labels': torch.int64, 'mask': torch.int64, 'boxes': torch.float32}
    print('batch:', type(batch), len(batch))
    print('batch[0]:', type(batch[0]), type(batch[0][0]))
    images = torch.stack([el['image'] for el in batch]).to(device)
    targets = [{
        key: torch.tensor(el[key], dtype=types[key], device=device) for key in el
        } for el in batch]
    return images, targets

tr = torch.utils.data.DataLoader(tr, 32, True, num_workers=4, pin_memory=True, collate_fn=collate_fn)

################################# DATA AUG #################################

soft_transform = v2.Compose([
    v2.Resize((ds.imgsize, ds.imgsize)),
    v2.RandomHorizontalFlip(),
    v2.ColorJitter(0.1, 0.1),
])
hard_transform = v2.AutoAugment()

################################## SETUP ##################################

model = getattr(models, ds.task.title())(ds.num_classes).to(device)
opt = torch.optim.Adam(model.parameters(), 1e-4)
method = getattr(methods, args.method)

################################## LOOP ##################################

model.train()
for epoch in range(args.epochs):
    tic = time()
    avg_sup_loss = 0
    avg_unsup_loss = 0
    for d, issup in tr:
        issup = issup.to(device)
        sup_loss = torch.mean(issup*model.forward_loss(d))
        unsup_loss = torch.mean((~issup)*method(model, d))
        loss = sup_loss + args.lamda*unsup_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        avg_sup_loss += float(sup_loss) / len(tr)
        avg_unsup_loss += float(unsup_loss) / len(tr)
    toc = time()
    print(f'Epoch {epoch+1}/{args.epochs} - {toc-tic:.0f}s - Avg loss: {avg_sup_loss} {avg_unsup_loss}')
