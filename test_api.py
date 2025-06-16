import requests

url = "http://127.0.0.1:5000/predict"
payload = {
    "input": [100, 120, 130, 140, 160, 180, 200],
    "n": 7
}

response = requests.post(url, json=payload)

print("Response:", response.json())  # 👈 ini WAJIB agar terlihat hasilnya
