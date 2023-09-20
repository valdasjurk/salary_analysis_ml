import requests

API = "http://localhost:8000/predict"

sample_tip_request_json = {
    "lytis": "M",
    "profesija": 722,
    "issilavinimas": "G4",
    "stazas": 10,
    "darbo_laiko_dalis": 100,
    "amzius": "30-39",
}

response = requests.post(API, json=sample_tip_request_json)
data = response.json()
print(data)
