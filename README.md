# Comparative Analysis of Image Classification Algorithms on the Fashion-MNIST Dataset

This repository contains all code, notebooks, models, datasets, and the full academic report for a comprehensive comparative study of image classification algorithms on the Fashion-MNIST dataset.
The project evaluates:
• Classical ML models → k-Nearest Neighbors (KNN), Logistic Regression, Support Vector Machines (SVM)
• Deep Learning model → Convolutional Neural Network (CNN)
• Modern Vision–Language model → CLIP (zero-shot classification)

The goal is to benchmark these methods on the same dataset and analyze their performance, strengths, and limitations.

---

This report includes:

- Abstract, Introduction, Literature Review
- Full Methodology (Data Processing, EDA, PCA, Model Training)
- CNN implementation details
- CLIP zero-shot evaluation
- Results, Business Insights, and Conclusion
- References in IEEE style

---

Repository Structure

fashion-mnist-comparative-analysis/
│
├── Comparative_Analysis_of_Image_Classification_Algorithms_on_Fashion_MNIST_Report.pdf
│
├── data/ # (Optional folder for external data if needed)
│
├── fashion_mnist_demo/ # CNN demo + scripts
│ ├── cnn_model.pth # Saved CNN model weights
│ ├── cnn_model.py # CNN architecture definition
│ ├── cnn_train.py # Training script used to train the CNN
││
├── fashion_mnist_train.csv # Training dataset (pre-converted)
├── fashion_mnist_val.csv # Validation dataset
├── fashion_mnist_test.csv # Test dataset
│
├── notebooks/ # Jupyter notebooks
│ ├── fashion_mnist_zero_shot.ipynb # CLIP zero-shot classification notebook
│ └── fashion_mnist_cnn_d_model.ipynb # CNN model development notebook
││
├── requirements.txt # Python dependencies
├── README.md # Project documentation (this file)
├── LICENSE # MIT License
└── .gitignore

---

Methods Implemented

1. Classical Machine Learning Models

All models are trained on PCA-reduced features (84 components):
• KNN: Hyperparameter tuning on k ∈ [1, 20]
• Logistic Regression: Max-iter tuned, multi-class softmax
• SVM: Grid search over kernels & cost parameters

These models operate on flattened grayscale pixel vectors.

2. Convolutional Neural Network (CNN)

A custom 3-layer CNN was implemented:
• Convolution → ReLU → MaxPooling
• Dropout for regularization
• Fully connected classifier head

The final model achieved:

- 92.07% test accuracy
- Highest precision, recall, and F1 among all supervised models
- Best confusion matrix performance

3. CLIP Zero-Shot Classification

The CLIP model (openai/clip-vit-base-patch16) was evaluated using handcrafted prompts:
• Zero-shot prediction without training
• Prompt engineering increases accuracy
• CLIP shows strong generalization but struggles on fine-grained classes (e.g., shirts vs t-shirts)

---

## Dataset

The Fashion-MNIST dataset contains 70,000 grayscale images of size 28x28 pixels, divided into 10 classes such as t-shirts, trousers, sneakers, etc.

| Class ID | Class Name  |
| -------- | ----------- |
| 0        | T-shirt/top |
| 1        | Trouser     |
| 2        | Pullover    |
| 3        | Dress       |
| 4        | Coat        |
| 5        | Sandal      |
| 6        | Shirt       |
| 7        | Sneaker     |
| 8        | Bag         |
| 9        | Ankle boot  |

---

## Evaluation Metrics

The models were evaluated using several metrics to provide a comprehensive comparison across the ten Fashion-MNIST classes:

### **1. Accuracy**

Overall percentage of correctly classified images.  
Used for baseline comparison across all algorithms.

### **2. Precision, Recall, and F1-Score**

Computed **per class** and averaged using **macro averaging**:

- **Precision:** Measures correctness of positive predictions
- **Recall:** Measures how many true samples were correctly identified
- **F1-Score:** Harmonic mean of precision and recall

These metrics reveal that certain classes (e.g., Shirt) are consistently harder to classify.

### **3. Confusion Matrix**

Confusion matrices were generated for:

- Logistic Regression
- KNN
- SVM
- CNN
- CLIP (zero-shot)

They highlight misclassification patterns and model-specific weaknesses.

### **4. ROC Curves and AUC (CNN)**

A multi-class One-vs-Rest ROC analysis was conducted for the CNN.  
All classes achieved **AUC ≥ 0.97**, showing strong separability and classifier confidence.

---

Results Summary
The following table summarizes the performance of all models evaluated in this study:

| **Model**                              | **Accuracy** | **Notes**                                                                                   |
| -------------------------------------- | ------------ | ------------------------------------------------------------------------------------------- |
| **k-Nearest Neighbors (KNN)**          | ~86%         | Serves as a simple baseline; sensitive to feature scaling and distance metrics.             |
| **Logistic Regression**                | ~84%         | Performs reasonably well but limited by linear separability.                                |
| **Support Vector Machine (SVM)**       | ~89%         | Best performance among classical ML models; effective with PCA-reduced features.            |
| **Convolutional Neural Network (CNN)** | **92.07%**   | Highest-performing model overall; excels at learning spatial features from raw pixel grids. |
| **CLIP (Zero-Shot Classification)**    | ~68%         | Strong generalization but struggles with Fashion-MNIST fine-grained classes.                |

---

## Project Structure

### Files:

1. **`cnn_model_train.py`**: Script for training the CNN and saving the trained model as `cnn_model.pth`.
2. **`cnn_main.py`**: Script for loading the saved model, evaluating it on the test dataset, and visualizing results.
3. **Fashion-MNIST Data**: Includes training, validation, and test datasets in CSV format.

### Outputs:

- **`cnn_model.pth`**: The saved weights of the trained CNN model.

---

## virtualenvironment

create a virtual environment to install the dependencies and libraries

```bash
virtualenv env_name
```

for MAC

1. Open your terminal.
2. Install virtualenv using pip:
   pip install virtualenv
3. Navigate to your project directory:
   cd /path/to/your/project
4. Create a virtual environment:
   virtualenv env_name
5. Run the following command to activate the virtual environment:
   source env_name/bin/activate

### Python Version:

- Python 3.7 or above

How to Run the Code

1. Clone the repository

git clone https://github.com/Shuvam-Chowdhury/fashion-mnist-comparative-analysis.git
cd fashion-mnist-comparative-analysis

2. Install dependencies

pip install -r requirements.txt

3. Run the CNN Demo

python cnn_train.py # Train the CNN model
python cnn_model.py # View model architecture

cnn_model.pth will be generated automatically.

4. Run Jupyter Notebooks

jupyter notebook

Then open:
• notebooks/fashion_mnist_cnn_d_model.ipynb
• notebooks/fashion_mnist_zero_shot.ipynb

License:
This project is released under the MIT License.

Acknowledgments:
Fashion-MNIST dataset by Zalando Research
CLIP model by OpenAI
Built with Python, PyTorch, NumPy, Pandas, Matplotlib, Scikit-Learn, and Hugging Face libraries.
