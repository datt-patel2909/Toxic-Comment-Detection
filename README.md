# 🛡️ Toxic Comment Detection — ML Mini Project

> **Hate Speech & Toxic Comment Detection** using Natural Language Processing and Machine Learning

## 📌 Overview

This project builds a complete ML pipeline to detect toxic comments using the **Jigsaw Toxic Comment Classification** dataset from Kaggle. It trains **4 different classifiers**, generates evaluation plots, and includes a **Streamlit web app** for live predictions.

## 🗂️ Project Structure

```
Toxic Comment Detection/
├── data/
│   ├── train.csv              
│   └── README.md
├── models/                   
├── outputs/                   
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py 
│   ├── model_training.py      
│   ├── evaluation.py         
│   └── predict.py             
├── app.py                     
├── main.py                   
├── requirements.txt
└── README.md
```

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.x | Programming language |
| Pandas & NumPy | Data manipulation |
| Scikit-learn | ML models & evaluation |
| NLTK | Text preprocessing & NLP |
| TF-IDF | Feature extraction |
| Matplotlib & Seaborn | Data visualization |
| Streamlit | Web application |
| Joblib | Model serialization |

## 🤖 Models Implemented

1. **Logistic Regression** — Fast, interpretable baseline
2. **Linear SVM** — Strong text classification performance
3. **Random Forest** — Ensemble-based approach
4. **Multinomial Naive Bayes** — Classic NLP classifier

## 📊 ML Pipeline

```
Raw Text → Cleaning → TF-IDF Vectorization → Model Training → Evaluation → Prediction
```

**Preprocessing steps:**
- Lowercasing
- URL & HTML tag removal
- Punctuation & digit removal
- Stopword removal
- Lemmatization
- TF-IDF with bigrams (50,000 features)

## 🚀 Setup & Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
### 2. Train Models
```bash
python main.py
```
This will:
- Load and clean the dataset
- Train all 4 models
- Generate evaluation plots in `outputs/`
- Print accuracy, precision, recall, and F1 scores

### 3. Launch Web App
```bash
streamlit run app.py
```
This opens a web interface where you can:
- Type any comment to check toxicity
- Choose between different ML models
- View training results and charts

### 4. Deploy to Web (Optional)
To host your project online for free so anyone can use it:
1. Push this project code to a public **GitHub repository**
2. Important: Make sure `models/*.pkl` and `requirements.txt` are pushed to GitHub!
   *(Note: The current `.gitignore` ignores `models/` because the files are large. To deploy, you must remove `models/*.pkl` from `.gitignore` first)*
3. Go to [share.streamlit.io](https://share.streamlit.io) and log in with GitHub
4. Click **New app**, select your repository, and set `app.py` as the main file path
5. Click **Deploy!**

## 📈 Evaluation Outputs

The pipeline generates these visualizations in `outputs/`:

| Chart | Description |
|---|---|
| `model_comparison.png` | Bar chart comparing all models |
| `roc_curves.png` | ROC curves with AUC scores |
| `confusion_matrices.png` | Confusion matrix for each model |
| `training_time.png` | Training time comparison |

## 👨‍🎓 Mini Project Details

- **Subject:** Machine Learning
- **Topic:** Hate Speech and Toxic Comment Detection
- **Dataset:** Jigsaw Toxic Comment Classification Challenge (Kaggle)
- **Approach:** Supervised Learning (Binary Classification)
