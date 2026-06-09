# ComplaintIQ 📊🤖
**Customer Complaint Analysis & Resolution Prediction Platform**

ComplaintIQ is an end-to-end Machine Learning pipeline and interactive web dashboard. It analyzes consumer complaints and uses a Random Forest Classifier to predict whether a new complaint will be resolved or remain unresolved, helping businesses prioritize customer service resources.

**🚀 Live Demo:** https://complaintiq-app.onrender.com/

---

## ✨ Features

- **Overview Dashboard**: High-level KPIs, priority distribution, issue frequency, and monthly volume trends.
- **Deep-Dive Analysis**: Visualizes resolution rates by product category, geographic state, and submission channel.
- **Model Performance**: Transparent model evaluation including a confusion matrix, feature importance ranking, and classification reports.
- **Prediction Engine**: Real-time inference API. Input complaint details to get an instant resolution prediction and confidence score.

## 🛠 Tech Stack
- **Backend:** Python, Flask, Gunicorn
- **Machine Learning:** Scikit-Learn, Pandas, NumPy
- **Frontend:** HTML5, CSS3, Chart.js

---

## 📁 Project Structure

```text
├── app.py                   # Flask web dashboard and API endpoints
├── generate_and_train.py    # Generates/cleans dataset & trains the Random Forest model
├── model.pkl                # Pre-trained RandomForest Classifier
├── stats.json               # Pre-computed analytics for rapid dashboard loading
├── complaints_dataset.csv   # Dataset used for model training
├── requirements.txt         # Python dependencies for local and production environments
└── .gitignore               # Git ignore rules