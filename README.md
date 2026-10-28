# Flower Image Classifier

An image classifier that recognizes five types of flowers and can also say **"not a flower"** when a photo shows something else. Built with transfer learning (MobileNetV2) in TensorFlow/Keras and deployed as an interactive demo with Gradio.

**Live demo:** [ADD YOUR HUGGING FACE SPACE LINK HERE]
**Author:** Tijani Ibrahim Gambari

![Demo screenshot](screenshots/demo.png)

---

## Problem

Given a photo, decide which kind of flower it shows: daisy, dandelion, rose, sunflower, or tulip. A basic classifier can only choose from the labels it was trained on, so it will always name a flower, even for a photo of a person or a car. To address this, I added a sixth class, `not_flower`, so the model can give an "none of these" answer.

## Dataset

| Class | Source |
|---|---|
| daisy, dandelion, roses, sunflowers, tulips | TensorFlow flowers dataset (about 3,670 photos) |
| not_flower | 700 random images from CIFAR-10 (animals, vehicles) plus [NUMBER] of my own photos of people, objects, and places |

- Split: 80% training, 20% validation (fixed seed so results are repeatable).
- Images resized to 224 x 224 pixels.

## Method

I used **transfer learning**: instead of training a network from scratch, I started from **MobileNetV2** pretrained on ImageNet and trained only a small new classification head on top.

1. **Base model:** MobileNetV2 (pretrained, frozen)
2. **Data augmentation:** random horizontal flip, rotation, zoom (reduces overfitting)
3. **Head:** Global average pooling, dropout (0.2), dense layer with 6 outputs
4. **Training:** Adam optimizer, sparse categorical cross-entropy loss, 5 epochs
5. **Prediction:** softmax converts outputs to probabilities for each class

Phone photos are stored with hidden rotation data, so uploaded images are corrected with `ImageOps.exif_transpose` before prediction.

## Results

| Metric | Value |
|---|---|
| Validation accuracy | [ADD YOUR NUMBER, e.g. 90.5%] |
| Training accuracy | [ADD YOUR NUMBER] |

![Training curves](screenshots/curves.png)

### Example predictions

| Photo | Prediction | Confidence | Correct? |
|---|---|---|---|
| [describe photo] | [class] | [XX%] | Yes / No |
| [describe photo] | [class] | [XX%] | Yes / No |
| [describe photo] | [class] | [XX%] | Yes / No |

![Example predictions](screenshots/predictions.png)

## What I learned

- **Transfer learning** gives strong results with little data and training time compared with training from scratch.
- **Always inspect your data first.** My first run loaded all images as a single class because of a folder path problem, which produced a meaningless 100% confidence. Checking the class names and counting classes caught this.
- **A perfect score is a warning sign,** not a success. It usually means a bug or data leak.
- **A classifier can only answer from its labels.** Without a `not_flower` class, the model confidently labeled non-flower photos as flowers.
- Phone images carry EXIF rotation metadata that must be handled explicitly.

## Limitations

- The model recognizes only five flower types. Other flowers may be mislabeled as one of them or as `not_flower`.
- The `not_flower` class only covers the kinds of non-flowers it was trained on. An unusual object could still be mistaken for a flower.
- Part of the `not_flower` data is low-resolution CIFAR-10 images, so the model may associate blurriness with "not a flower." Adding more of my own high-quality photos reduces this.
- Accuracy was measured on one validation split and 5 epochs of training, with no extensive tuning.

## Future improvements

- Fine-tune the top layers of MobileNetV2 for higher accuracy.
- Add more flower classes and more varied `not_flower` photos.
- Add a confusion matrix to see which classes get mixed up.
- Calibrate confidence so low-confidence predictions return "not sure."

## How to run it

1. Open `flower_classifier.ipynb` in [Google Colab](https://colab.research.google.com).
2. Set the runtime to GPU: **Runtime > Change runtime type > T4 GPU**.
3. Run the cell. It downloads the data, trains the model, and launches a Gradio demo with a public link.

To run the saved model locally:

```bash
pip install tensorflow gradio pillow numpy
python app.py
```

(`app.py` and `flower_classifier.keras` must be in the same folder.)

## Tech stack

Python, TensorFlow/Keras, MobileNetV2, NumPy, Pillow, Matplotlib, Gradio, Google Colab

## Repository structure

```
.
├── flower_classifier.ipynb   # full notebook
├── app.py                    # Gradio app
├── requirements.txt
├── screenshots/
│   ├── demo.png
│   ├── curves.png
│   └── predictions.png
└── README.md
```
