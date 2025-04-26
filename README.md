**AI-Driven Crime Pattern Prediction and Interactive Visualization Using Socioeconomic and POI Data**

**📚 Overview**

This project develops an AI-powered system to detect crime patterns and predict future incidents using large-scale Chicago crime data, socioeconomic indicators, and Points of Interest (POIs).
It integrates machine learning (KNN, Random Forest, XGBoost) and deep learning (CNN-LSTM) models with an interactive Streamlit web application for real-time crime prediction and exploration.

**🚀 Project Objectives**

Detect and predict crime patterns using spatial, temporal, and socioeconomic data.

Assess the impact of POIs (schools, parks, libraries, vacant buildings, police stations) and Hardship Index on crime predictions.

Deploy an interactive Streamlit web application for user-friendly crime analysis and visualization.

**🧰 Tools & Technologies**

**Category	                           |  Technologies Used**
Programming Languages                  |	Python (Pandas, NumPy, Scikit-learn, PyTorch)
Machine Learning & Deep Learning	     | KNN, Random Forest, XGBoost, CNN-LSTM
Visualization	                         | Matplotlib, Seaborn, SHAP, LIME, Streamlit
Model Tuning & Evaluation	             | GridSearchCV, RandomizedSearchCV, ROC-AUC, F1-Score
Data Handling & Processing	           | Z-score outlier removal, feature engineering, PCA
Deployment	                           | Streamlit Web App
Version Control	                       | Git, GitHub

**📊 Key Features**

Data Cleaning & Feature Engineering: Outlier removal, crime type reclassification, datetime extraction, POI merging, and socioeconomic integration.

Automated Feature Selection: Mutual Information, RFE, Sequential Feature Selection, and PCA.

Advanced Model Training:

ML Models: KNN, Random Forest, XGBoost with hyperparameter optimization.

DL Model: CNN-LSTM hybrid for spatial-temporal pattern recognition.

Model Interpretability: Local and global explanations using SHAP, LIME, and Permutation Importance.

Performance Metrics: Accuracy, F1-score, Recall, Precision, Confusion Matrix, and ROC-AUC.

Interactive Streamlit App: Explore crime predictions, feature impacts, and dynamic visualizations.


**📈 Sample Visualizations**

Crime trend heatmaps
<img width="367" alt="image" src="https://github.com/user-attachments/assets/b70fef0e-cb3c-4e70-a60a-bac1aee414c1" />


Confusion matrices for model evaluation


<img width="611" alt="image" src="https://github.com/user-attachments/assets/51ce745d-1569-4c0a-a349-5b3c07b1fa61" />


SHAP summary plots showing feature importance

<img width="477" alt="image" src="https://github.com/user-attachments/assets/29476053-b711-480f-9bf5-bd9938b537c9" />


Real-time crime prediction dashboard (Streamlit)

**🧠 Future Improvements**

Integrate Kernel Density Estimation (KDE) for hotspot detection.

Apply Geographically Weighted Regression (GWR) for deeper spatial analysis.

Expand model interpretability using counterfactual explanations.

Enhance deployment via Docker and AWS/GCP hosting.

**🤝 Acknowledgements**

Chicago Data Portal for open-access crime, POI, and socioeconomic data.

Research studies guiding crime type reclassification and feature selection methods.

📬 Contact
Kehinde Ogundana
[LinkedIn](http://www.linkedin.com/in/kehindeogundana) | Email: ogundanakehinde2022@gmail.com
