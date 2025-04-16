# Codeforces Algorithm Tag Classifier

This project aims to classify Codeforces problems into algorithmic tags based on their description, editorial notes, and source code. It's a multi-label classification problem and serves as a good example of a text-based machine learning pipeline.

---

## Getting started

First, clone the repository and set up your environment. Run it either in linux either in MacOS.

```bash
git clone <your-repo-url>
cd <your-repo-folder>
python -m venv venv
source venv/bin/activate 
pip install -r requirements.txt
```

⸻

Dataset

The dataset you should use is code_classification_dataset_v3.csv.
This file was generated as part of the exploratory data analysis (EDA) and can be found in the notebooks folder.
You’ll need to pass the path to this CSV when training the model.

Once you trigger the training pipeline, the data is split into training and test sets automatically, and the test set is saved in:
```bash
data/test_data.csv
```
You don’t need to manually split the data.

⸻

How to train the model

There are two ways to train the model: with or without cross-validation.

Standard training (without CV)
```bash
python main.py train \
  --csv_path notebooks/code_classification_dataset_v3.csv \
  --model_path models/model.pkl \
  --vectorizer_path models/vectorizer.pkl \
  --binarizer_path models/binarizer.pkl
```
This will:
	•	Train the model on 80% of the data (train/test split is done automatically)
	•	Save the test set as data/test_data.csv for later evaluation
	•	Save the model and its components in the models/ folder

With cross-validation (for hyperparameter tuning)
```bash
python main.py train \
  --csv_path notebooks/code_classification_dataset_v3.csv \
  --model_path models/model.pkl \
  --vectorizer_path models/vectorizer.pkl \
  --binarizer_path models/binarizer.pkl \
  --cv
```
This will do a grid search over TF-IDF + Logistic Regression hyperparameters and train the final model using the best config.

⸻

How to evaluate the model

Once you’ve trained the model, you can evaluate its performance on the test set:
```bash
python main.py evaluate \
  --test_csv_path data/test_data.csv \
  --model_path models/model.pkl \
  --vectorizer_path models/vectorizer.pkl \
  --binarizer_path models/binarizer.pkl \
  --save_reports
```
This will print classification reports for:
	•	All tags
	•	Focus tags only (a set of 8 tags that are particularly important)

It will also save the evaluation reports to the reports/ folder.

⸻

Focus Tags

The model is trained to predict all tags that appear in the dataset. However, we pay special attention to a list of focus tags during evaluation:

['games', 'geometry', 'graphs', 'math', 'number theory', 'probabilities', 'strings', 'trees']


⸻

What’s inside
	•	train.py: trains the model and saves artifacts
	•	evaluate.py: loads the model and generates evaluation metrics
	•	main.py: CLI wrapper to run training or evaluation
	•	utils.py: preprocessing functions
	•	config.py: configuration constants like random seed and focus tag list
	•	models/: where models and transformers are stored
	•	data/: where the test split is saved
	•	notebooks/: contains EDA and the final dataset (code_classification_dataset_v3.csv)
    •	reports/: contains the metrics on the test set. 

⸻

Requirements

Everything you need is in requirements.txt. You can install it with:
```bash
pip install -r requirements.txt
```
⸻

Metrics

Here are the final metrics obtained using the default configuration (TF-IDF + Logistic Regression):

Classification Report — All Tags

Metric	Precision	Recall	F1-score	Support
Micro avg	0.45	0.55	0.49	2778
Macro avg	0.44	0.54	0.48	2778
Weighted avg	0.46	0.55	0.50	2778
Samples avg	0.47	0.58	0.48	2778
Hamming Loss	—	—	—	0.1228

Hamming loss represents the fraction of incorrect labels. Lower is better.

⸻


Here’s the full classification report for the focus tags, formatted for direct copy-paste into your README or other documents:

⸻

Classification Report — Focus Tags Only

               precision    recall  f1-score   support

        games       0.57      0.81      0.67        21
     geometry       0.56      0.68      0.61        34
       graphs       0.59      0.65      0.62       133
         math       0.51      0.60      0.55       267
number theory       0.52      0.66      0.58        65
probabilities       0.71      0.55      0.62        22
      strings       0.61      0.77      0.68        87
        trees       0.49      0.75      0.59        59

    micro avg       0.54      0.66      0.60       688
    macro avg       0.57      0.68      0.61       688
 weighted avg       0.55      0.66      0.59       688
  samples avg       0.33      0.36      0.33       688

Hamming Loss (Focus Tags): 0.0779

This performance shows consistent results across key tags like math, graphs, and strings, with micro F1-score of 0.60 and low Hamming Loss, which reflects strong multi-label accuracy for the most relevant categories.

⸻

Final notes
	•	The project is structured to make it easy to extend (e.g. semantic embeddings, new models, API).
	•	You can re-run the EDA notebook to generate or inspect the dataset.
	•	Training and evaluation run fully from the CLI.
⸻
