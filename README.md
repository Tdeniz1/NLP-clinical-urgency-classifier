# Healthcare NLP Classifier – Urgency Detection from Clinical Notes

This project builds a custom NLP pipeline to classify clinical text notes into urgency levels: **EMERGENCY**, **URGENT**, **ROUTINE**, and **NON-URGENT**. It uses synthetic, HIPAA-compliant data to simulate a real-world clinical triage support system.

This tool is built to accept **any CSV dataset**, no matter the number of columns or rows. It automatically removes empty columns and rows, isolates the relevant text column, cleans it, and classifies each entry into one of four urgency levels using machine learning. No special formatting or structure is required. Just a CSV file with a column containing clinical text.


------------------------------------------------------------------------------------------------------------------------------------------------

## Key Takeaways

- Built a full NLP classification pipeline from scratch, including data cleaning, feature engineering, model training, and evaluation
- Implemented a composite scoring framework to select statistically robust models using both test and cross-validation metrics
- Engineered a flexible, column-agnostic pipeline capable of handling real-world messy datasets—critical in applied AI workflows
- Demonstrated practical experience with text vectorization, label encoding, and performance tradeoff analysis
- Gained hands-on exposure to model interpretability challenges in NLP—an essential skill in real-world AI applications
- Designed the project structure for modularity and future deployment (e.g., as an API or Streamlit app), reflecting industry best practices


------------------------------------------------------------------------------------------------------------------------------------------------

## Features

- Preprocessing using **NLTK** and **spaCy**
- Label encoding for multi-class classification
- TF-IDF vectorization
- Models tested: **Multinomial Naive Bayes**, **SGD Classifier**, **Random Forest**, and **Logistic Regression**
- Evaluation using accuracy, confusion matrix, and classification report
- Clean, modular codebase for reuse and scalability

------------------------------------------------------------------------------------------------------------------------------------------------

## Model Performance

- **Best Accuracy Achieved**: `85.00%` test accuracy  
- **Cross-Validation Accuracy**: `78.75% ± 4.54%`  
- **Top Performing Model**: `Multinomial Naive Bayes` (based on composite scoring of accuracy and CV stats)
  
------------------------------------------------------------------------------------------------------------------------------------------------

## Dataset

Since real medical data is protected under HIPAA and not publicly available, OpenAI was used to generate a synthetic dataset for training all models in this project.

- **Source**: Custom synthetic dataset
- **Format**: CSV with `text` and `label` columns
- **Classes**: EMERGENCY, URGENT, ROUTINE, NON-URGENT
- This dataset contains **no real patient data**.

------------------------------------------------------------------------------------------------------------------------------------------------

## Requirements

Install all required dependencies using:

```bash
pip install -r requirements.txt
```

Main libraries used:
- `pandas`
- `scikit-learn`
- `nltk`
- `spacy`
- `tqdm`

------------------------------------------------------------------------------------------------------------------------------------------------

## Usage

Follow these steps to run the classifier from start to finish:

# 1. Load the dataset
import pandas as pd
df = pd.read_csv("**your data set CSV goes here**")  
# ⤷ Loads the synthetic clinical notes into a pandas DataFrame

# 2. Clean the text
from src.preprocessing import clean_text
df['clean_text'] = df['text'].apply(clean_text)  
# ⤷ Applies custom preprocessing:
#    - Lowercasing
#    - Removing punctuation and non-letter characters
#    - Removing NLTK stopwords

# 3. Train the model
from src.train_model import train
train(df)  
# ⤷ Trains multiple models using TF-IDF features
# ⤷ Evaluates test and cross-validation accuracy
# ⤷ Calculates a composite score to rank models statistically
# ⤷ Prints the best-performing model based on accuracy + stability

# 4. Make a prediction on a new clinical note
from src.predict import predict_urgency
predict_urgency("Patient reports chest pain and shortness of breath.")  
# ⤷ Cleans and vectorizes the input text
# ⤷ Uses the best trained model to predict the urgency class:
#    EMERGENCY, URGENT, ROUTINE, or NON-URGENT


------------------------------------------------------------------------------------------------------------------------------------------------

## Sample Output

```
Input Text:  "Patient reports chest pain and shortness of breath."
Predicted Urgency:  EMERGENCY
```

------------------------------------------------------------------------------------------------------------------------------------------------

## Real-World Use

This classifier simulates the core logic of a triage support tool used in hospitals or digital health platforms. In a production setting, it could:

- Automatically flag urgent clinical notes from large volumes of incoming text
- Prioritize patient cases based on urgency (e.g., EMERGENCY vs NON-URGENT)
- Support clinicians, nurses, or telehealth platforms in real-time decision-making
- Be integrated into EHR systems or patient intake apps to improve workflow efficiency

While this project uses synthetic data, the pipeline design mirrors what would be required for deployment on real HIPAA-compliant clinical data after proper validation.

------------------------------------------------------------------------------------------------------------------------------------------------

## Future Improvements

- Streamlit deployment for interactive prediction
- Integration with BERT or other transformer models
- Model explainability with SHAP or LIME
- This project serves as a foundation for more advanced work in clinical NLP, including transformer-based models (e.g., BERT) and production-grade model deployment.

------------------------------------------------------------------------------------------------------------------------------------------------

## Engineering Highlights

- **Composite scoring system** combining test and cross-validation metrics to select the statistically best model
- Custom `clean_text()` preprocessing pipeline built from scratch using regex, NLTK, and spaCy
- Dynamically loads and filters any dataset shape (column-agnostic design)
- Designed with reproducibility and modularity in mind for easy handoff or deployment

