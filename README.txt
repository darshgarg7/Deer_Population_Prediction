Deer Population Prediction Project
This project provides a robust pipeline for predicting deer population trends using machine learning. By leveraging historical environmental and population data, it implements advanced feature engineering, hyperparameter optimization, and model evaluation techniques. The project aims to deliver accurate, interpretable predictions that can inform wildlife management strategies.

🚀 Key Features
End-to-End Pipeline:
Data preprocessing with advanced techniques such as imputation, feature scaling, and polynomial feature expansion.
Model implementation using Random Forest Regressor, XGBoost, and custom LSTM neural networks for enhanced predictive accuracy.
Feature Engineering:
Inclusion of statistical lags, rolling means, standard deviations, interaction terms, and polynomial features to improve model performance.
Hyperparameter Optimization:
Utilizes Optuna for exhaustive hyperparameter tuning, ensuring the best-performing models.
Model Evaluation:
Performance metrics include Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R², Mean Absolute Error (MAE), and Mean Absolute Percentage Error (MAPE).
Experiment Tracking:
Integrated with MLflow for tracking experiments, parameters, and models, ensuring reproducibility and experiment comparability.
Model Interpretability:
SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations) are used for visualizing and understanding model predictions.
Cross-Validation:
Implemented K-Fold cross-validation to ensure robust, generalizable model performance.

📂 Project Structure
Deer_Population_Prediction/
├── population_predictor.py          # Core script for model training and prediction
├── final_data.csv                   # Historical population and environmental data
├── test_model.py                    # Test the population prediction model
├── README.md                        # Project documentation
├── devcontainer/                    # Devcontainer
├─────────── Dockerfile              # Containerizing the application
├─────────── gitlab-ci_cd.yml        # CI/CD pipeline for team
├─────────── requirements.txt        # List of Python dependencies
└── models/                          # Folder for storing trained models

🛠️ Tools & Technologies
Programming Languages: Python 3.x
Libraries:
Machine Learning: Scikit-learn, Optuna, SHAP
Data Manipulation: Pandas, NumPy
Visualization: Matplotlib, Seaborn
Experiment Tracking: MLflow
Modeling Techniques: Random Forest Regressor, Gradient Boosting
Development Environment: Jupyter Notebook, CLI, Docker (for containerization)

📊 Results
The model achieved an R² score of 0.85, indicating a strong fit to the data.
Root Mean Squared Error (RMSE): 25.123, demonstrating a reliable level of accuracy.
Hyperparameter optimization improved model performance by 15% over baseline results.
💡 How to Run
Clone the repository:

git clone https://github.com/darshgarg7/Deer_Population_Prediction.git
cd Deer_Population_Prediction

Install dependencies (optional):

pip install -r requirements.txt

Run Docker ([locally] - run CI/CD pipeline for development):

docker build -t Deer_Population_Prediction
docker run Deer_Population_Prediction

To train the model and generate predictions:

python3 population_predictor.py

Run tests (optional):

pytest test_model.py

📈 Visualizations
Actual vs. Predicted Population: Visual comparison of real versus predicted values to assess model accuracy.

Feature Importance: SHAP-based visualizations that highlight the most influential features in the model's decision-making process.

Contributors
Darsh Garg - Scrum Master & Technical Lead
Fostered an environment of continuous collaboration through daily stand-ups and sprint retrospectives. Iteratively refined the machine learning pipeline through feedback loops, leading the evolution from Random Forest to LSTM and XGBoost implementations. Facilitated knowledge sharing sessions and pair programming to strengthen team capabilities.
Core Development
Rohan Dham - Analytics & Visualization Engineer
Worked closely with the technical lead to develop interactive visualizations that enhanced model interpretability. Collaborated on the Random Forest foundation through paired programming sessions, enabling rapid prototyping and validation of approaches. Regularly shared insights with the team to drive data-driven decisions.
Data Engineering
Chinmay Goyal - Data Pipeline Specialist
Established robust data pipelines while maintaining continuous communication with modeling teams to understand evolving data needs. Implemented automated quality checks and participated in daily scrums to ensure data readiness aligned with sprint goals. Regularly collaborated with QA to validate data integrity.
Quality & Testing
Keshav Goyal - Quality Assurance Engineer
Partnered closely with all team members to implement comprehensive testing strategies throughout the development lifecycle. Provided rapid feedback on model performance and data quality, enabling quick iterations and improvements. Maintained testing documentation and facilitated testing-focused knowledge sharing sessions.

📊 Dataset Overview
The dataset contains historical data on deer populations, harvests, and environmental factors like temperature, precipitation, and snow depth. The table below outlines the key variables:

# https://docs.google.com/spreadsheets/d/1vhPbtmKqaRHKEAuw4RfHTYIrzuOnJAavmAAIPz9g_rA/edit?usp=sharing
#   Big thank you to the Minnesota Department of Agriculture for providing nessesary informtion!

🔧 Future Improvements
Model Refinement: Experiment with other models such as LightGBM for performance comparison.
Additional Features: Incorporate more environmental variables (e.g., soil moisture, vegetation data) to improve accuracy.
Deployment: Develop a REST API for real-time predictions, allowing wildlife managers to input current environmental data for future population projections. 

Contact:

Darsh Garg: darsh.garg@gmail.com & garg0088@umn.edu

Rohan Dham: rohand400@gmail.com & dham0014@umn.edu

Chinmay Goyal: goyal127@umn.edu

Keshav Gautam: gautakes000@gmail.com
