# Lithuanian salary analysis 2018

data source: https://get.data.gov.lt/datasets/gov/lsd/darbo_uzmokestis/DarboUzmokestis2018 

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



## Machine learning WEB API with FastAPI:

```bash
python train.py
```
```bash
python -m uvicorn app:app --reload
```
```bash
python test_request.py
```
You can also try out to predict the salary virtualy on: http://localhost:8000/docs




