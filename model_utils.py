# model_utils.py
import torch
from torch import nn
from torch.optim import lr_scheduler
from torchvision import models, datasets, transforms
from torch.autograd import Variable

def build_model(arch='vgg16', hidden_units=4096):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        raise ValueError("Unsupported architecture")

    for param in model.parameters():
        param.requires_grad = False

    classifier_input_size = model.classifier[0].in_features
    classifier = nn.Sequential(
        nn.Linear(classifier_input_size, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_units, 102),  
        nn.LogSoftmax(dim=1)
    )

    model.classifier = classifier
    model.class_to_idx = None  

    return model

def train_model(model, dataloaders, dataset_sizes, learning_rate, epochs, use_gpu, save_dir):
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    for epoch in range(epochs):
        print(f'Epoch {epoch}/{epochs - 1}')

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                if use_gpu and torch.cuda.is_available():
                    inputs, labels = inputs.cuda(), labels.cuda()
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    save_checkpoint(model, save_dir)

def save_checkpoint(model, save_dir):
    checkpoint = {
        'arch': 'vgg16',  # Adjust accordingly based on the actual architecture used
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }
    torch.save(checkpoint, f'{save_dir}/checkpoint.pth')

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif checkpoint['arch'] == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        raise ValueError("Unsupported architecture")
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model

def predict(model, image, top_k=5, use_gpu=False):
    model.eval()
    if use_gpu and torch.cuda.is_available():
        model.cuda()
        image = image.cuda()
    else:
        model.cpu()
    image = Variable(image.unsqueeze(0))
    output = model.forward(image)
    ps = torch.exp(output)
    top_p, top_class = ps.topk(top_k, dim=1)
    class_to_idx_inv = {i: c for c, i in model.class_to_idx.items()}
    classes = [class_to_idx_inv[idx] for idx in top_class.cpu().numpy()[0]]
    return top_p.cpu().numpy()[0], classes
