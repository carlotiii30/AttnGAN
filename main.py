import os

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from torch.utils.data import DataLoader

from src.dataset import CocoDataset
from src.discriminator import Discriminator
from src.generator import Generator
from src.text_encoder import TextEncoder
from src.vocabulary import Vocabulary

# Parámetros
BATCH_SIZE = 64
Z_DIM = 100
TEXT_DIM = 256
IMAGE_CHANNELS = 3
NDF = 64
NGF = 64
LEARNING_RATE_G = 0.0001
LEARNING_RATE_D = 0.0004
BETA1 = 0.5
NUM_EPOCHS = 200

# Rutas
ANNOTATION_FILE = "data/coco/annotations/captions_train2017.json"
IMAGE_DIR = "data/coco/train2017"
OUTPUT_DIR = "data/output"


# Función de colación personalizada
def collate_fn(batch):
    images, batch_captions = zip(*batch)

    batch_captions = [caption.clone().detach() for caption in batch_captions]
    captions_padded = torch.nn.utils.rnn.pad_sequence(batch_captions, batch_first=True)

    images = torch.stack(images, 0)
    return images, captions_padded


# Usar GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando el dispositivo: {device}")

# Inicializar las transformaciones para las imágenes
transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# Crear el vocabulario
vocab = Vocabulary()

# Cargar el dataset de COCO
coco = COCO(ANNOTATION_FILE)

# Recorrer todas las captions en el dataset y añadir palabras al vocabulario
for annotation_id in coco.anns:
    caption = coco.anns[annotation_id]["caption"]
    palabras = caption.lower().split()
    for palabra in palabras:
        vocab.add_word(palabra)

# Crear el dataset de COCO
dataset = CocoDataset(
    image_dir=IMAGE_DIR,
    annotation_file=ANNOTATION_FILE,
    transform=transform,
    vocab=vocab,
)
print(f"Número de imágenes en el dataset: {len(dataset)}")

# Crear el DataLoader
dataloader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)
print(f"Número de batches: {len(dataloader)}")

# Inicializar el generador, discriminador y text encoder
text_encoder = TextEncoder(len(vocab), embedding_dim=256, hidden_size=256).to(device)
generator = Generator(Z_DIM, TEXT_DIM, IMAGE_CHANNELS, NGF).to(device)
discriminator = Discriminator(IMAGE_CHANNELS, TEXT_DIM, NDF).to(device)

# Definir la función de pérdida
criterion = torch.nn.BCELoss()

# Definir los optimizadores
optimizer_g = optim.Adam(
    generator.parameters(), lr=LEARNING_RATE_G, betas=(BETA1, 0.999)
)
optimizer_d = optim.Adam(
    discriminator.parameters(), lr=LEARNING_RATE_D, betas=(BETA1, 0.999)
)

# Bucle de entrenamiento
num_batches = len(dataloader) - 200

for epoch in range(NUM_EPOCHS):
    for i, data in enumerate(dataloader):
        if i >= num_batches:
            break

        real_images, captions = data

        real_images = real_images.to(device)
        captions = captions.to(device)

        text_embeddings = text_encoder(captions)

        real_labels = torch.ones((BATCH_SIZE, 1, 1, 1)).to(device)
        fake_labels = torch.zeros((BATCH_SIZE, 1, 1, 1)).to(device)

        ######################
        # Actualizar el Discriminador
        ######################
        discriminator.zero_grad()

        # Pérdida del discriminador con imágenes reales
        real_output = discriminator(real_images, text_embeddings)
        d_real_loss = criterion(real_output, real_labels.expand_as(real_output))

        # Generar imágenes falsas
        noise = torch.randn(BATCH_SIZE, Z_DIM).to(device)
        fake_images = generator(noise, text_embeddings)

        # Pérdida del discriminador con imágenes falsas
        fake_output = discriminator(fake_images.detach(), text_embeddings)
        d_fake_loss = criterion(fake_output, fake_labels.expand_as(fake_output))

        d_loss = d_real_loss + d_fake_loss
        d_loss.backward(retain_graph=True)
        optimizer_d.step()

        ######################
        # Actualizar el Generador
        ######################
        generator.zero_grad()

        # El generador intenta engañar al discriminador
        output = discriminator(fake_images, text_embeddings)
        g_loss = criterion(output, real_labels.expand_as(output))

        g_loss.backward()
        optimizer_g.step()

        if i % 100 == 0:
            print(
                f"Epoch {epoch}/{NUM_EPOCHS}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}"
            )

    # Guardar los modelos y las imágenes generadas
    if epoch % 10 == 0:
        print(f"Guardando los modelos en el epoch {epoch}")
        torch.save(
            generator.state_dict(), f"data/output/models/generator_epoch_{epoch}.pth"
        )
        torch.save(
            discriminator.state_dict(),
            os.path.join(OUTPUT_DIR, f"models/discriminator_epoch_{epoch}.pth"),
        )

        # Guardar imagen generada
        print("Guardando imagen generada")
        noise = torch.randn(1, Z_DIM).to(device)
        with torch.no_grad():
            text_embedding_example = text_encoder(captions[0].unsqueeze(0))
            sample_image = (
                generator(noise, text_embedding_example)
                .cpu()
                .squeeze()
                .permute(1, 2, 0)
                .numpy()
            )

            plt.imshow((sample_image + 1) / 2)
            plt.axis("off")
            plt.savefig(
                os.path.join(OUTPUT_DIR, f"images/generated_image_epoch_{epoch}.png"),
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()
