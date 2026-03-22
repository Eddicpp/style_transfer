import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# 1. Definiamo le trasformazioni per VGG-19
# Le medie e deviazioni standard sono quelle usate su ImageNet
vgg_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225])

def load_image(img_path, max_size=400, shape=None):
    """ Carica e prepara l'immagine per la rete. """
    image = Image.open(img_path).convert('RGB')
    
    # Ridimensioniamo per non far esplodere la memoria del Mac
    size = max_size if max(image.size) > max_size else max(image.size)
    if shape is not None:
        size = shape
        
    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        vgg_normalization  # Fondamentale!
    ])

    # Aggiungiamo la dimensione del "batch" (YOLO ne voleva 16, qui ne basta 1)
    image = in_transform(image).unsqueeze(0)
    return image

def im_convert(tensor):
    """ Converte un tensore in un'immagine visualizzabile (denormalizzazione). """
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0) # Da (C, H, W) a (H, W, C)
    
    # Invertiamo la normalizzazione di prima
    image = image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    image = image.clip(0, 1) # Assicuriamoci che i colori siano tra 0 e 1
    return image