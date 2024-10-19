## Estructura
|- data
|   |- coco
|   |   |- annotations
|   |   |- train2017
|   |
|   |- output
|   |   |- images
|   |   |- models
|
|- src
|   |- dataset.py
|   |- discriminator.py
|   |- generator.py
|   |- text_encoder.py
|   |- vocabulary.py
|
|- main.py


## Instrucciones

1. Copia las carpetas ``train2017`` y ``annotations`` de COCO en ``data/models``.

2. Prepara el entorno: ``poetry install``

3. Ejecuta la aplicación: ``poetry run python main.py``
