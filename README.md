# Classification-of-Autistic-Children

Binary image classifier (`autistic` / `non_autistic`) for children's facial images, built with PyTorch using a ResNet backbone.

## Structure
- `train.py` – trains the model (Adam, lr `1e-4`, CrossEntropy) and saves the checkpoint and loss/accuracy plots
- `inference.py` – predicts the class of a single image and saves the annotated result to `output/`
- `datasets.py` – data loaders and augmentations (batch size 32, 224px)
- `resnet.py`, `custom_resnet.py`, `modules/` – model definitions
- `utils.py` – helpers to save the model and plots

## Usage
1. Download the dataset from Kaggle and extract it into `data/` (`data/train`, `data/valid`, `data/test`).
2. Train:
   ```bash
   python train.py --epochs 200
   ```
3. Predict:
   ```bash
   python inference.py --input data/test/autistic/001.jpg
   ```
   Update the checkpoint path in `inference.py` to point to your trained model.
