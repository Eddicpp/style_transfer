import torch
import torch.optim as optim
from utils import load_image, im_convert
from vgg_model import StyleExtractor, get_gram_matrix
import matplotlib.pyplot as plt

style_weights = {
    'conv1_1': 1.0, 
    'conv2_1': 0.8, 
    'conv3_1': 0.5, 
    'conv4_1': 0.3, 
    'conv5_1': 0.1
}

# Alpha (Contenuto) e Beta (Stile)
content_weight = 1   # Quanto vuoi che si veda la foto originale
style_weight = 1e6   # Quanto vuoi che si veda il quadro (di solito un numero molto alto)

# 1. Configurazione Device (Sfruttiamo il tuo Mac!)
device = torch.device("cpu")
print(f"🚀 Usando il dispositivo: {device}")

# 2. Carichiamo le immagini
# Sostituisci i nomi dei file con i tuoi!
content = load_image('images/content/content.jpeg', max_size=200).to(device)
# Ridimensioniamo lo stile per farlo combaciare col contenuto
style = load_image('images/style/style.jpeg', shape=content.shape[-2:]).to(device)

# 3. Prepariamo l'immagine Target
# Partiamo dalla foto del leone e diciamo a PyTorch: "Voglio calcolare i gradienti sui PIXEL"
target = content.clone().requires_grad_(True).to(device)

# 4. Inizializziamo l'estrattore
extractor = StyleExtractor().to(device)

print(f"DEBUG: Target device: {target.device} | Shape: {target.shape}")
print(f"DEBUG: Extractor device: {next(extractor.parameters()).device}")
print(f"DEBUG: Extractor training mode: {extractor.training}")

# 5. Calcoliamo le caratteristiche fisse (NON serve il gradiente qui!)
with torch.no_grad():
    content_features = extractor(content)
    style_features = extractor(style)
    style_grams = {layer: get_gram_matrix(style_features[layer]) for layer in style_features}

# 6. Pesi e Ottimizzatore (abbassiamo un po' il tiro per stabilità)
optimizer = optim.Adam([target], lr=0.02) # LR un po' più alto per vedere subito i cambiamenti

print("🎨 Inizio della pittura neurale...")

# 8. Ciclo di ottimizzazione
steps = 20000
for i in range(1, steps + 1):
    # Estraiamo le feature dell'immagine che stiamo modificando
    target_features = extractor(target)
    
    # --- CALCOLO CONTENT LOSS ---
    # Confrontiamo il layer profondo conv4_2
    c_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
    
    # --- CALCOLO STYLE LOSS ---
    s_loss = 0
    for layer in style_weights:
        # Prendi le caratteristiche del layer corrente
        target_feature = target_features[layer]
        target_gram = get_gram_matrix(target_feature)
        style_gram = style_grams[layer]
        
        # Calcoliamo l'errore tra le due matrici di Gram
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
        
        # NORMALIZZAZIONE CORRETTA: 
        # Usiamo il numero di elementi (C * H * W) del tensore originale
        _, c, h, w = target_feature.shape
        s_loss += layer_style_loss / (c * h * w)

    # --- TOTAL LOSS ---
    total_loss = content_weight * c_loss + style_weight * s_loss
    
    # Backpropagation: calcola come cambiare i pixel
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    if i % 100 == 0:
        print(f"Step {i}/{steps} | Visualizzazione evoluzione...")
        plt.imshow(im_convert(target))
        plt.axis('off')
        plt.pause(0.1) # Mostra l'immagine per 0.1 secondi e continua
        # Salviamo anche una copia intermedia per sicurezza
        plt.imsave(f'outputs/step_{i}.png', im_convert(target))

# Salvataggio Finale
final_img = im_convert(target)
plt.imsave('outputs/capolavoro_finale.png', final_img)
print("✅ Quadro completato e salvato in 'outputs/'!")