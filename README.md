# Disaster Response Pipeline Project

## Project Overview
In the wake of natural disasters, emergency-response organizations are inundated with tweets, text messages, and service‑desk tickets.  
Manually triaging these requests costs precious minutes that could save lives.

This repo delivers an **end‑to‑end pipeline**—from raw CSV files to a production‑ready Flask app—that automatically classifies each incoming message into 36 humanitarian‑response categories (e.g., *water*, *search and rescue*, *medical help*), ensuring they reach the right teams fast.

The system combines standard NLP preprocessing (tokenization, lemmatization, TF‑IDF) with a grid‑searched **multi‑output RandomForest/AdaBoost ensemble**.  
Performance is tracked with accuracy, precision, recall, and F1, and the trained model is exposed through a simple web UI so field operators can paste a message and instantly see which relief teams should respond.

![Application Interface](resources/Disaster_Response_Application_Interface.png)
![Classification Result](resources/Disaster_Response_Classification_Result.png)

---

## Technical Architecture

```
Raw CSV ─► ETL Pipeline ─► SQLite DB ─► ML Pipeline ─► Pickled Model ─► Flask App
```

### Interactive Jupyter Notebooks

| Notebook | Purpose |
|----------|---------|
| [ETL_Pipeline_Preparation.ipynb](resources/ETL_Pipeline_Preparation.ipynb) | Step‑by‑step exploration of the data‑cleaning workflow implemented in `process_data.py`. |
| [ML_Pipeline_Preparation.ipynb](resources/ML_Pipeline_Preparation.ipynb) | Interactive training, tuning, and evaluation of the multi‑output classifier (mirrors `train_classifier.py`). |

> **Tip :** No Jupyter? GitHub renders notebooks automatically.

---

## 1  ETL Pipeline (Extract → Transform → Load)

`process_data.py`:

* Reads message and category CSVs.  
* Merges datasets, fixes inconsistencies, removes duplicates.  
* One‑hot‑encodes the 36 category columns.  
* Saves the clean result to `DisasterResponse.db` (SQLite).

---

## 2  Machine‑Learning Pipeline

`train_classifier.py`:

* Splits data, builds a `Pipeline` (TF‑IDF → classifier).  
* Runs `GridSearchCV` over key hyper‑parameters (`n_estimators`, `max_depth`, etc.).  
* Prints classification report per label and saves the best model to `classifier.pkl`.

Test‑set highlights:

* **Accuracy :** 94.4 %  
* **Weighted F1 :** 0.86  
* Solid recall on frequent labels (*related*, *request*); room for improvement on rare ones (*refugees*, *shops*).

---

## Getting Started

1. **Clone the repo**

   ```bash
   git clone https://github.com/Soriano-R/disaster-response-pipeline.git
   cd disaster-response-pipeline
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Build the database**

   ```bash
   python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
   ```

4. **Train the model**

   ```bash
   python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
   ```

5. **Run the web app**

   ```bash
   cd app
   python run.py
   ```

   Open <http://localhost:3001> in your browser.

---

## Repository Structure

```
├── app/
│   ├── run.py
│   └── templates/
│       ├── master.html
│       └── go.html
│
├── data/
│   ├── disaster_messages.csv
│   ├── disaster_categories.csv
│   ├── process_data.py
│   └── DisasterResponse.db         ← generated
│
├── models/
│   ├── train_classifier.py
│   └── classifier.pkl              ← generated
│
├── resources/
│   ├── Disaster_Response_Application_Interface.png
│   ├── Disaster_Response_Classification_Result.png
│   ├── ETL_Pipeline_Preparation.ipynb
│   ├── ML_Pipeline_Preparation.ipynb
│   ├── ETL_Pipeline_Preparation.html
│   └── ML_Pipeline_Preparation.html
│
├── requirements.txt
└── README.md
```

---

## License
This project is released under the MIT License. See `LICENSE` for details.

