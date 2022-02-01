from torchvision.models import resnet50
import torch
import torch.nn as nn
import os
from PIL import Image
import numpy as np
import torchvision.transforms as TF
from pathlib import Path
from collections import Counter

def get_filelist(pt):
    with open(pt, 'r') as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    files = [l.split()[0] for l in lines]
    labels = [int(l.split()[1]) for l in lines]
    return files, labels

def read_image(pt):
    _normalize_tfm = TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    _to_tensfor_transform = TF.Compose([TF.ToTensor(), _normalize_tfm])

    img = Image.open(pt)
    img = TF.Resize((256, 256))(img)
    img = np.array(img)
    if len(img.shape) == 2:
        img = (TF.ToTensor()(img) - 0.5) / 0.5
        img = img.repeat(3,1,1)
    else:
        img = _to_tensfor_transform(img)
    return img

def main():
    DATA_FOLDER = '/mnt/tank/scratch/tpolevaya/datasets/office/'

    model = resnet50(pretrained=True)
    layers = list(model.children())[:-1] + [nn.Flatten()]
    encoder = nn.Sequential(*layers).eval().cuda()

    FOLDER = Path('data_handling/features')
    FOLDER.mkdir(exist_ok=False)

    for domain in ['art', 'clipart', 'product', 'real_world']:
        pt = f'data_handling/filelists/{domain}_0-64_test.txt'
        files, labels = get_filelist(pt)
        files = [DATA_FOLDER + f[len('office-home/'):] for f in files]
        print(domain, len(files))
        features = []
        for f in files:
            img = read_image(f)
            with torch.no_grad():
                out = encoder(img.unsqueeze(0).cuda()).squeeze(0)
            features.append(out.detach().cpu())
        features = torch.stack(features, dim=0)
        labels = torch.tensor(labels)
        torch.save(features, FOLDER / f'OH_{domain}_features.pt')
        torch.save(labels, FOLDER / f'OH_{domain}_labels.pt')

if __name__ == '__main__':
    main()
