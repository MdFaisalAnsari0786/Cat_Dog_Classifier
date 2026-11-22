# Cat vs Dog Image Classification

A semester mini-project for **Deep Learning for Computer Vision**. It uses a Convolutional Neural Network (CNN) with **MobileNetV2 transfer learning** to classify an uploaded image as a cat or a dog.

## Project features

- Downloads the Cats vs Dogs image dataset from the Microsoft-hosted archive used by TensorFlow Datasets.
- Removes unreadable/corrupt images and creates train, validation and test folders (80/10/10 split).
- Uses data augmentation (horizontal flip, rotation and zoom).
- Uses a pre-trained MobileNetV2 convolutional base for feature extraction.
- Optionally fine-tunes the last 30 layers.
- Evaluates on a held-out test split and saves metrics and an accuracy plot.
- Provides a Streamlit web app to upload an image and get a Cat/Dog prediction.

## Requirements

- Windows 10/11, macOS or Linux (64-bit)
- Python **3.10–3.12** recommended
- Internet access for downloading the dataset and MobileNetV2 weights
- About 2–3 GB free disk space recommended for dataset, environment and model files

TensorFlow's official installation guide currently lists Python 3.9–3.12 for its tested pip packages. See: https://www.tensorflow.org/install/

## Setup on Windows (VS Code / Command Prompt)

1. Extract this project ZIP.
2. Open the `Cat_Dog_Classifier` folder in VS Code.
3. Open Terminal and create a virtual environment (this project setup is written for the Python 3.10 installation detected on your laptop):

   ```bash
   py -3.10 -m venv .venv
   .venv\Scripts\activate
   ```

4. Install libraries:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. Download and prepare the dataset (around 787 MiB):

   ```bash
   python download_data.py
   ```

6. Train the CNN transfer-learning model:

   ```bash
   python train.py
   ```

7. Start the image-prediction web app only after training has completed and `models/cat_dog_model.keras` exists:

   ```bash
   streamlit run app.py
   ```

**Important on Windows:** Run each command separately and wait for it to finish before entering the next one. Do not paste `python download_data.py python train.py streamlit run app.py` as one command. The correct order is `python download_data.py`, then `python train.py`, and finally `streamlit run app.py`.

   Streamlit will print a local address (usually `http://localhost:8501`). Open it in your browser, upload a JPG/PNG image, and click **Predict image**.

## Optional fine-tuning

After the classification head has been trained, fine-tune the last MobileNetV2 layers:

```bash
python train.py --epochs 5 --fine-tune --fine-tune-epochs 3
```

You can reduce the batch size if your computer runs out of memory:

```bash
python train.py --batch-size 16 --epochs 3
```

## Project structure

```text
Cat_Dog_Classifier/
├── app.py                  # Streamlit prediction interface
├── train.py                # CNN model training and evaluation
├── download_data.py        # Dataset download, validation and splitting
├── requirements.txt        # Python dependencies
├── project_report.md       # Report draft for semester submission
├── viva_questions.md       # Viva questions with answers
├── data/
│   ├── raw/                # Downloaded archive and extracted images
│   ├── train/cats, train/dogs
│   ├── val/cats, val/dogs
│   └── test/cats, test/dogs
├── models/                 # Saved trained model and class names
└── outputs/                # Metrics, CSV logs and training plot
```

## How the model works

1. An uploaded image is resized to 160 × 160 pixels.
2. Data augmentation is applied during training only.
3. MobileNetV2 converts the image into visual feature maps.
4. Global Average Pooling and a sigmoid output layer predict the dog probability.
5. The application displays the class with the higher probability.

The folder names are sorted as `cats`, `dogs`; therefore the model's sigmoid output represents the probability of the `dogs` class.

## Important notes

- A trained model is **not included** in this starter ZIP. Run `download_data.py` and `train.py` on your computer to create `models/cat_dog_model.keras`.
- Do not write a guessed accuracy in the report. Copy the actual `test_accuracy` value from `outputs/metrics.json` after training.
- Training time depends on your computer. CPU training can take significantly longer than GPU training.
- The model is an educational classifier, not a veterinary or safety system.

## Dataset and references

- TensorFlow Datasets – Cats vs Dogs catalogue: https://www.tensorflow.org/datasets/catalog/cats_vs_dogs
- Dataset source/archive URL is defined in `download_data.py`; TensorFlow Datasets describes 23,262 usable examples and 1,738 corrupted images skipped.
- TensorFlow transfer learning tutorial: https://www.tensorflow.org/tutorials/images/transfer_learning
- TensorFlow installation guide: https://www.tensorflow.org/install/

## Easiest way on Windows (one-click)

Open the `Cat_Dog_Classifier` folder and double-click `START_PROJECT_WINDOWS.bat`. It creates the Python 3.10 environment if needed, installs the packages, prepares the dataset if it is not already prepared, trains the model if the model file is missing, and then launches Streamlit. Keep the window open. The first setup downloads about 787 MiB of image data and training may take a while. If the window displays an error, send a screenshot of the error text.
"# Cat_Dog_Classifier" 
