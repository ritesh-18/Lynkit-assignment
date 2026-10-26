# License Plate Recognition with PaddleOCR — PP-OCRv5 vs PP-OCRv4

Two Google Colab notebooks that fine-tune and compare PaddleOCR text-recognition models on car license plates.

| Notebook | What it does |
|---|---|
| [`PPOCR_License_Plate_Training.ipynb`](PPOCR_License_Plate_Training.ipynb) | Dataset preparation → fine-tune `PP-OCRv5_server_rec` and `PP-OCRv4_server_rec` → training curves → evaluation (zero-shot vs fine-tuned) → export to inference format → comparison |
| [`PPOCR_License_Plate_Inference.ipynb`](PPOCR_License_Plate_Inference.ipynb) | Load both fine-tuned models → detection + recognition on full car photos → annotated images (boxes, text, confidence) → side-by-side comparison → batch processing → error analysis → export |

## Approach

```
car photo ──► PP-OCR text detection (pretrained) ──► plate box ──► fine-tuned recognizer ──► plate text + confidence
```

The assignment's models (`PP-OCRv5_server_rec`, `PP-OCRv4_server_rec`) are **recognition** models: they read the characters in a cropped text image.
They are fine-tuned on cropped plates, starting from the official pretrained weights.
For full car photos, each version's pretrained detector (`PP-OCRv5_server_det` / `PP-OCRv4_server_det`) finds the text boxes first.

## Dataset

| Dataset | Source | License | Used for |
|---|---|---|---|
| European License Plates | [Kaggle: abdelhamidzakaria/european-license-plates-dataset](https://www.kaggle.com/datasets/abdelhamidzakaria/european-license-plates-dataset) | MIT | training / validation / held-out test |
| Car License Plate Detection | [Kaggle: andrewmvd/car-plate-detection](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection) | CC0 | full-scene inference (433 photos with plate boxes) |

**Preparation methodology** (implemented in the training notebook, section 3):
1. Download with the Kaggle API.
2. The label is the file name. Duplicate-photo suffixes (`_1`, `_2`) are stripped, and labels are checked to contain only `A–Z0–9`.
3. Kaggle's train and val folders are merged and re-split **80/20 by plate number**, so photos of the same plate never appear in both. Result: 530 train / 132 val images.
4. The original test folder is kept as a held-out test set. 6 images whose plate also appears in train/val are removed, leaving 67.
5. Labels are written in PaddleOCR format: `relative/path.png<TAB>LABEL`.

## How to run (Google Colab)

1. Open the notebook in Colab and select `Runtime → Change runtime type → T4 GPU`.
2. Add your Kaggle API token as a Colab secret named **`KAGGLE_API_TOKEN`** (key icon in the left sidebar), and enable notebook access.
3. Run **`PPOCR_License_Plate_Training.ipynb`** with `Runtime → Run all`. Authorize Google Drive when asked. The trained models are saved to `MyDrive/ppocr_license_plate/`.
4. Run **`PPOCR_License_Plate_Inference.ipynb`** with `Runtime → Run all`. It loads the models from the same Drive folder.

Pinned versions: `paddlepaddle-gpu==3.3.1` (CUDA 12.6), `paddleocr==3.7.0`, and PaddleOCR repo tag `v3.7.0`.

### Running locally (optional)
```bash
pip install paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
pip install -r requirements.txt
cp .env.example .env   # then paste your Kaggle token
```
Outputs go to `work/` and `artifacts/` instead of Google Drive.

## Outputs

Saved to `MyDrive/ppocr_license_plate/`:
```
inference/PP-OCRv5_server_rec/, inference/PP-OCRv4_server_rec/   exported models
configs/                                                          training configs used
results/                                                          training logs, curves, evaluation CSVs, comparison plots
test_data/                                                        held-out test crops + label files
inference_outputs/                                                annotated images, batch CSVs, error-analysis plots
```

## Results

Both notebooks were run end-to-end on a local NVIDIA RTX 3050 Laptop GPU (6 GB) with 40 epochs, batch size 32 and learning rate 5e-5. All outputs are saved in the notebooks.

| Model | Test accuracy, zero-shot → fine-tuned | Test accuracy, all 67 plates | Char accuracy | Plate detection rate (car photos) | Full pipeline latency | Size |
|---|---|---|---|---|---|---|
| PP-OCRv5_server_rec | 29.7% → 96.9% | 95.5% | 98.1% | 81.5% | 40.8 ms/image | 85 MB |
| PP-OCRv4_server_rec | 28.1% → 96.9% | 95.5% | 98.1% | 82.0% | 47.0 ms/image | 91 MB |

The zero-shot → fine-tuned figures come from PaddleOCR's `eval.py`, which scored 64 of the 67 test plates. The all-67 column comes from the exported models.

**Key findings**
- Fine-tuning on 530 plates takes both models from under 30% to about 96% exact-match accuracy.
- PP-OCRv5 and PP-OCRv4 are **tied** on accuracy: both get the same 3 test plates wrong. PP-OCRv5 is slightly smaller and faster, so it is the recommended model.
- A learning rate of 2e-4 overfit the small dataset. 5e-5 worked for both models.
- On real car photos, the general-purpose text **detector** is the bottleneck. It misses about 18% of plates and cuts or merges another ~31%, while recognition of cropped plates is about 96% accurate. The next step would be fine-tuning the detector on plate boxes.

See section 12 of the training notebook and section 10 of the inference notebook for the full analysis.
