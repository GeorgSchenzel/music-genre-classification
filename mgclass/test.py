import requests

resp = requests.post("http://localhost:5000/predict",
                     files={"file": open("/home/georg/Music/ADL/wav_16k/3Form - Goodbye.wav", 'rb')})

print(resp)
print(resp.json())
