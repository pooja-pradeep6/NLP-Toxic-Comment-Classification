NLP-Based Toxic Comment Classification with Explainability

## Project Overview

This mini project uses Natural Language Processing (NLP) and Machine Learning to automatically classify online comments as **Toxic** or **Clean**.

The project also focuses on **explainability**. LIME (Local Interpretable Model-agnostic Explanations) is used to explain individual predictions by showing which words contributed to the model's decision.

The main goal is not only to identify potentially toxic comments, but also to make the model's predictions easier to understand.

## Problem Statement

Online platforms receive a large number of comments, making manual moderation difficult. This project develops an NLP-based text classification system that automatically identifies whether a comment is toxic or clean.

The project additionally uses LIME to provide a word-level explanation for individual predictions.

## Objectives

* Classify online comments as Toxic or Clean.
* Apply NLP preprocessing techniques to text data.
* Convert text into numerical features using TF-IDF.
* Train a Logistic Regression classification model.
* Evaluate the model using Accuracy, Precision, Recall, F1-score, and Confusion Matrix.
* Explain individual predictions using LIME.

## Dataset

The project uses the **Civil Comments** dataset, loaded directly from Hugging Face Datasets.

The dataset contains online comments along with a `toxicity` score between 0 and 1. For this project, a sample of **30,000 comments** was selected using a fixed random state of 42.

A binary target label was created:

* `0` → Clean
* `1` → Toxic

A comment is considered Toxic when its toxicity score is **greater than or equal to 0.5**.

### Working Dataset

* Total comments: **30,000**
* Clean comments: **27,655**
* Toxic comments: **2,345**
* Toxic comments: **7.82%**

## Methodology

The project follows this workflow:

**Text Data → Preprocessing → TF-IDF → Logistic Regression → Prediction → Evaluation → LIME Explanation**

### 1. Text Preprocessing

The comments are cleaned before classification. The preprocessing includes:

* Converting text to lowercase
* Removing URLs
* Removing newline characters
* Removing punctuation and numbers
* Removing English stopwords
* Removing very short words

### 2. Train-Test Split

The dataset is divided into:

* **80% training data**
* **20% testing data**

Stratified splitting is used to maintain the proportion of Toxic and Clean comments in both sets.

### 3. TF-IDF

TF-IDF (Term Frequency-Inverse Document Frequency) is used to convert the cleaned text into numerical features.

The vectorizer is configured with a maximum of **5,000 features**.

### 4. Logistic Regression

Logistic Regression is used as the classification model.

It was selected because it works well with high-dimensional sparse TF-IDF features, is computationally efficient, and is relatively easy to understand and explain.

The model uses `class_weight='balanced'` to give more importance to the minority Toxic class.

### 5. LIME Explainability

LIME is used to explain individual predictions.

For a given comment, LIME identifies words that contribute positively or negatively to the model's prediction. This helps understand why the model classified a particular comment as Toxic or Clean.

## Results

The model achieved an overall accuracy of approximately **87%** on the test set.

| Class | Precision | Recall | F1-score |
| ----- | --------: | -----: | -------: |
| Clean |      0.96 |   0.90 |     0.93 |
| Toxic |      0.33 |   0.58 |     0.42 |

Overall:

* Accuracy: **0.87**
* Macro F1-score: **0.67**
* Weighted F1-score: **0.89**

Because the dataset is imbalanced, accuracy alone does not fully describe the model's performance. The Toxic class has a recall of **0.58**, meaning the model identifies a reasonable portion of toxic comments, although some toxic comments are still missed.

## Explainability

LIME was applied to sample comments to understand individual predictions.

For example, the project tests the model with custom comments such as:

> "You are so stupid, nobody wants to hear your opinion"

The model predicts this comment as **Toxic**, and LIME provides an explanation showing the words that influenced the prediction.

## Limitations

The project has some limitations:

* The working dataset is a 30,000-comment sample rather than the complete dataset.
* Toxic comments are much fewer than Clean comments, resulting in class imbalance.
* Sarcasm and context can be difficult for the model to understand.
* Removing punctuation and numbers during preprocessing may remove some useful information.
* The Logistic Regression model may not capture complex language patterns as effectively as advanced language models.
* Predictions on unseen or context-dependent comments may sometimes be incorrect.

## Future Scope

The project can be improved by:

* Using a larger portion of the dataset.
* Experimenting with models such as Linear SVM or transformer-based models.
* Improving text preprocessing.
* Supporting multilingual comments.
* Developing a real-time toxic comment detection application.
* Deploying the model as a web or mobile application.
* Exploring additional explainability techniques.

## Technologies Used

* Python
* Pandas
* NumPy
* NLTK
* Scikit-learn
* Matplotlib
* Hugging Face Datasets
* LIME

## How to Run

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
cd NLP-Toxic-Comment-Classification
```

### 2. Install the required libraries

```bash
pip install -r requirements.txt
```

### 3. Open the notebook

Open `NLP_Project.ipynb` using Google Colab or Jupyter Notebook.

### 4. Run the notebook

Run the cells in order to:

1. Load the Civil Comments dataset.
2. Prepare the binary labels.
3. Preprocess the comments.
4. Split the dataset.
5. Generate TF-IDF features.
6. Train the Logistic Regression model.
7. Evaluate the model.
8. Generate LIME explanations.

## Project Files

* `NLP_Project.ipynb` – Complete Python implementation with outputs.
* `README.md` – Project description and instructions.
* `requirements.txt` – Required Python libraries.
* `report/` – Project report.
* `screenshots/` – Important result screenshots.

## Author

**Pooja Pradeep**

BCA – Artificial Intelligence & Machine Learning
