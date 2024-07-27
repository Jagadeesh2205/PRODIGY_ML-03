
# Cats vs Dogs Image Classification using SVM

This repository contains code to classify images of cats and dogs using a Support Vector Machine (SVM) model. The dataset used is the [Kaggle Cats and Dogs dataset](https://www.kaggle.com/c/dogs-vs-cats/data).

## Requirements

- Python 3.x
- numpy
- scikit-learn
- Pillow
- tqdm

You can install the required packages using:


```markdown
pip install numpy scikit-learn Pillow tqdm
```

## Dataset

Download the dataset from the Kaggle competition page: [Kaggle Cats and Dogs dataset](https://www.kaggle.com/c/dogs-vs-cats/data). Extract the contents of `train.zip` and `test1.zip` into appropriate directories.

## Project Structure

```
.
├── train                  # Directory containing training images
│   ├── cat.0.jpg
│   ├── cat.1.jpg
│   ├── ...
│   └── dog.12499.jpg
├── test1                  # Directory containing test images
│   ├── 1.jpg
│   ├── 2.jpg
│   ├── ...
│   └── 12500.jpg
├── svm_cats_vs_dogs.py    # Main code to train and test SVM model
└── README.md              # This readme file
```

## Usage

1. **Set up directories**: Ensure your `train` and `test1` directories are in place with the respective images.

2. **Run the script**:
```bash
python svm_cats_vs_dogs.py
```

The script will:
- Load and preprocess the images.
- Extract features from the images.
- Train an SVM model on the training data.
- Validate the model on a validation set.
- Predict labels for the test set and print them.

### Code Explanation

- `load_images_from_folder`: Function to load images from a directory, resize them, and assign labels.
- `extract_features`: Function to flatten image arrays into feature vectors.
- Main script:
  - Loads and preprocesses training images.
  - Splits the data into training and validation sets.
  - Trains an SVM model.
  - Validates the model and prints the validation accuracy.
  - Loads and preprocesses test images.
  - Predicts labels for the test images and prints them.

## Notes

- Adjust the `train_dir` and `test_dir` variables in the script to point to your actual data directories.
- The `batch_size` can be adjusted based on your system's memory capacity.
- The SVM model uses a linear kernel for simplicity. You can experiment with different kernels and hyperparameters for potentially better performance.

