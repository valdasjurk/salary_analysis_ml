# Lithuanian salary analysis 2018

data source: https://get.data.gov.lt/datasets/gov/lsd/darbo_uzmokestis/DarboUzmokestis2018 
profession codes list: https://www.profesijuklasifikatorius.lt/?q=lt/medziosarasas

Project structure:
```
project/
├── data/
│ ├── raw/
├── src/
│ ├── predictions/
│ ├── preprocess/
│ ├── visualization/
├── README.md
└── requirements.txt
└── run.py
```
## ML model feature importances
![Model](https://github.com/valdasjurk/Airplane_accidents_analysis/blob/cee1910b0d895ab2efe8d828a7185eca81d4e3ec/Figure_32.png)

## Install required modules
```bash
pip install -r requirements.txt
```
## Script main functions

Create scikit LinearRegression model and get score:
```bash
python run.py --create_lr_model_and_show_score 
```
Create scikit RandomForestRegression model and get score:
```bash
python run.py --create_rfr_model_and_show_score 
```
Compare scikit LinearRegression and pyTorch Linear models:
```bash
python run.py --compare_lr_scikit_to_torch_by_mse
```
Create various testing scenarios with scikit LinearRegression model. Function takes arguments: experience_year (from, to), profession code name, age group and education (G2, G4) and results plotting or saving (--show True for plot, False for saving)(given values as example):
```bash
python run.py --create_testing_scenarios_and_predict --experience_year 1,31 --profession 251 --age_group 30-39 --education G4 --show True
```
You can predict yearly salary with an input of: sex, age group, profession code, work experience, workload and education degree (given values as example).
```bash
python run.py --predict_yearly_salary --sex M --age 30-39 --profession_code 334 --exp 5 --workload 100 --educ G4
```
Plot or save model feature imporances with SHAP. Function takes one argument (--show) for plotting (True) or saving (False):
```bash
python run.py --shap_feature_importances --show True
```

## Machine learning WEB API with FastAPI:

```bash
python FastAPI/train.py
```
```bash
python -m uvicorn app:app --reload
```
```bash
python FastAPI/test_request.py
```
You can also try out to predict the salary virtualy on: http://localhost:8000/docs




