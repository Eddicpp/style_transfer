import torch
from torchvision import models

class StyleExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Carichiamo VGG19
        vgg = models.vgg19(weights='DEFAULT').features
        
        # --- FIX FONDAMENTALE ---
        # Disattiviamo tutti gli 'inplace=True' che fanno crashare la memoria su Mac
        for layer in vgg.modules():
            if isinstance(layer, torch.nn.ReLU):
                layer.inplace = False
        
        vgg.eval()
        for param in vgg.parameters():
            param.requires_grad_(False)
            
        self.vgg = vgg
        self.layers = {
            '0': 'conv1_1', '5': 'conv2_1', 
            '10': 'conv3_1', '19': 'conv4_1', 
            '21': 'conv4_2', '28': 'conv5_1'
        }

    def forward(self, x):
        features = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.layers:
                # Usiamo .clone() per essere sicuri di non avere conflitti in memoria
                features[self.layers[name]] = x.clone()
        return features

def get_gram_matrix(tensor):
    # Assicuriamoci che il tensore sia float32
    tensor = tensor.float()
    b, c, h, w = tensor.size()
    tensor = tensor.view(b * c, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram