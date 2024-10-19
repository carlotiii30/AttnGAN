import os

from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset

from src.vocabulary import tokenize_caption


class CocoDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None, vocab=None):
        self.image_dir = image_dir
        self.transform = transform
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
        self.vocab = vocab

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        # Obtener el ID de la imagen
        image_id = self.image_ids[index]

        # Cargar la imagen
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.image_dir, image_info["file_name"])
        image = Image.open(image_path).convert("RGB")

        # Obtener el caption
        annotation_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(annotation_ids)
        caption = annotations[0]["caption"]  # Usar la primera anotación

        # Tokenizar el caption
        if self.vocab:
            caption = tokenize_caption(caption, self.vocab)

        if self.transform is not None:
            image = self.transform(image)

        return image, caption
