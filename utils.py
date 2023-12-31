# utils.py
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.models import vgg16, densenet121
from PIL import Image
from workspace_utils import active_session

def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test']),
    }

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid': DataLoader(image_datasets['valid'], batch_size=32),
        'test': DataLoader(image_datasets['test'], batch_size=32),
    }
    return dataloaders['train'], dataloaders['valid'], dataloaders['test']

def save_checkpoint(model, save_dir):
    model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = {
        'arch': 'vgg16',
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }
    with active_session():
        torch.save(checkpoint, f'{save_dir}/checkpoint.pth')

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if checkpoint['arch'] == 'vgg16':
        model = vgg16(pretrained=True)
    elif checkpoint['arch'] == 'densenet121':
        model = densenet121(pretrained=True)
    else:
        raise ValueError("Unsupported architecture")
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model

def process_image(image_path):
    img = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img_tensor = preprocess(img)
    return img_tensor
