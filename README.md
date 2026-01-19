# Estimated Time of Arrival (ETA) Prediction
![App (1)](https://github.com/Azie88/Estimated-Time-of-Arrival-ETA-Prediction/assets/101363399/62edada4-6a6a-415a-9770-774859666d41)

This project implements a machine learning solution to predict the **Estimated Time of Arrival (ETA)** for a single journey in a ride-hailing app context.  

---

## Gradio App Huggingface Link

| Project Name | Deployed App |
|------|------|
| ETA prediction | <a href="https://huggingface.co/spaces/Azie88/ETA_Prediction_App" target="_blank">Gradio App on Huggingface</a> |

---

## Executive Summary of Results

This project evaluated multiple machine learning models to predict Estimated Time of Arrival using historical trip data. Among the tested approaches, gradient-boosted tree models consistently outperformed traditional regression and distance-based methods. The final XGBoost model achieved the lowest prediction error, with an RMSE of **152.90**, which further improved to **151.75** after hyperparameter tuning. These results indicate that the model is able to accurately capture complex, non-linear relationships between trip characteristics and travel time, making it suitable for real-world ETA prediction scenarios where consistency and robustness are critical.

---

## Problem Statement

Accurately predicting arrival times is a critical challenge in ride-hailing and logistics platforms.  
Travel duration is influenced by multiple factors, including distance, geography, and routing patterns.  
Inaccurate ETAs lead to poor user experience, inefficient fleet management, and increased operational costs.

This project frames ETA estimation as a **supervised regression problem** and applies machine learning to learn patterns from historical trip data.

---

## Solution Approach

The project follows a structured end-to-end machine learning workflow:

1. **Data Understanding & Cleaning**
   - Validation of geographic coordinates
   - Handling missing and inconsistent values

2. **Exploratory Data Analysis (EDA)**
   - Univariate and bivariate analysis
   - Spatial visualization of origin and destination points
   - Statistical hypothesis testing

3. **Feature Engineering**
   - Use of origin and destination coordinates
   - Derived trip-level features
   - Preparation of data for modeling

4. **Modeling & Evaluation**
   - Multiple regression models evaluated
   - Proper train/validation/test split
   - Comparison using standard regression metrics

5. **Deployment**
   - Model wrapped in a Gradio application
   - Deployed publicly via Hugging Face Spaces

---

## Model Performance & Evaluation

The models were evaluated using **Root Mean Squared Error (RMSE)**, which penalizes larger prediction errors and is well-suited for ETA prediction tasks.

### Model Comparison Results

| Model | RMSE (Lower is Better) |
|------|------------------------|
| **XGBoost (XGB)** | **152.904** |
| LightGBM (LGBM) | 170.359 |
| Linear Regression | 217.904 |
| K-Nearest Neighbors (KNN) | 236.419 |
| Random Forest | 241.029 |
| Decision Tree | 256.356 |

### Interpretation

- XGBoost achieved the lowest RMSE, indicating the highest overall prediction accuracy.
- Ensemble tree-based models significantly outperformed simpler and more rigid models.
- Higher RMSE values suggest reduced robustness, particularly for longer or more complex trips.

In practical terms, an RMSE of ~153 means the model’s predictions deviate from the true ETA by approximately that amount (depending on the dataset’s time scale), with fewer extreme errors compared to other models.

---

### Final Model Selection

**XGBoost was selected as the final model** due to its superior performance and robustness.

After hyperparameter tuning, the XGBoost model achieved:

- **Tuned XGBoost RMSE:** **151.75**

This confirms that careful parameter optimization can further improve predictive accuracy beyond the baseline model.

---

## Why XGBoost?

XGBoost was selected as the final model because it offers an effective balance between performance, flexibility, and real-world applicability.

From a technical perspective:
- It captures **non-linear relationships** better than linear models
- It handles **feature interactions** automatically
- It is robust to noise and outliers common in operational trip data

From a business and stakeholder perspective:
- It delivers **more reliable ETAs**, reducing extreme under- or over-estimations
- It scales well to larger datasets
- It is widely adopted in industry, making it a proven and trusted solution

These characteristics make XGBoost a strong choice for production-oriented ETA prediction systems.

---

## Project Structure 📂

- `Dataset/`: Contains the dataset used for analysis, and predicted values.
- `Dev/`: Contains jupyter notebook with full end-to-end ML process
- `toolkit/`: Pipeline with ML model
- `.gitignore`: Holds files to be ignored by Git.
- `app.py`: Working Gradio app for prediction
- `LICENSE`: Project license.
- `README.md`: Project overview, links, highlights, and information.
- `requirements.txt`: Required libraries & packages

## Getting Started🏁

You need to have [`Python 3`](https://www.python.org/) on your system. Then you can clone this repo and being at the repo's `root :: repository_name> ...`

1. Clone this repository: `git clone https://github.com/Azie88/Estimated-Time-of-Arrival-ETA-Prediction.git`
2. On your IDE, create A Virtual Environment and Install the required packages for the project:

- Windows:
        
        python -m venv venv; 
        venv\Scripts\activate; 
        python -m pip install -q --upgrade pip; 
        python -m pip install -qr requirements.txt  

- Linux & MacOs:
        
        python3 -m venv venv; 
        source venv/bin/activate; 
        python -m pip install -q --upgrade pip; 
        python -m pip install -qr requirements.txt  

The two long command-lines have the same structure. They pipe multiple commands using the symbol ` ; ` but you can manually execute them one after the other.

- **Create the Python's virtual environment** that isolates the required libraries of the project to avoid conflicts;
- **Activate the Python's virtual environment** so that the Python kernel & libraries will be those of the isolated environment;
- **Upgrade Pip, the installed libraries/packages manager** to have the up-to-date version that will work correctly;
- **Install the required libraries/packages** listed in the `requirements.txt` file so that they can be imported into the python script and notebook without any issue.

**NB:** For MacOs users, please install `Xcode` if you have an issue.

- Run the Gradio app (being at the repository root):

  Gradio: 
  
    For development

      gradio app.py
    
    For normal deployment/execution

      python app.py  

  - Go to your browser at the following address :
        
      http://localhost:7860

3. Explore the Jupyter notebook for detailed steps and code execution.
4. Check out the live running app on [Huggingface Spaces](https://huggingface.co/spaces/Azie88/ETA_Prediction_App).

## Gradio App Screenshots

![Gradio App 1](https://github.com/Azie88/Estimated-Time-of-Arrival-ETA-Prediction/assets/101363399/2208e2eb-d271-40b6-a495-1880ce7f2e50)
![Gradio App  2](https://github.com/Azie88/Estimated-Time-of-Arrival-ETA-Prediction/assets/101363399/92ffdbc7-545e-4d3c-b2db-5731d0fdce6c)



## Author✍️

Andrew Obando

Connect with me on LinkedIn: [Andrew Obando](https://www.linkedin.com/in/andrewobando/)

---

Feel free to star ⭐ this repository if you find it helpful!
