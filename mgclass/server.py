import torch
import torchaudio
from flask import Flask, jsonify, request, current_app
from miniaudio import SampleFormat, decode
import urllib.request
from io import BytesIO

from mgclass import networks
from mgclass.utils import create_spectrogram

print("Loading model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = networks.MgcNet(8)
data = urllib.request.urlopen("https://github.com/GeorgSchenzel/music-genre-classification/releases/download/model-v1/MgcNet.pt")
model.load_state_dict(torch.load(BytesIO(data.read()), map_location=device)["model_state_dict"])
model.eval()

class_labels = ["Deep House",
                "DnB",
                "Future Rave",
                "House Classic",
                "Liquid DnB",
                "Psytrance",
                "Tech House",
                "Techno"]

mean = 31.1943
std = 293.4358

print("Finished loading model")


def create_input_from_bytes(audio_bytes):
    decoded_audio = decode(audio_bytes, nchannels=1, sample_rate=16000, output_format=SampleFormat.SIGNED32)
    data = torch.FloatTensor(decoded_audio.samples)[None, None, :]

    # normalize 32 integer bit audio by dividing by 2147483648 (or short hand 1 << 31)
    data /= (1 << 31)

    data = create_spectrogram(win_length=2048)(data)
    data = (data - mean) / std

    return data


def inference(data):
    num_classes = 8

    count = 0
    pred = torch.zeros((1, num_classes))

    for i in range(0, data.shape[3] - 128, 128):
        count += 1
        pred += model(data[:, :, :, i: i + 128])

    pred /= count

    _, y_hat = pred.max(1)

    return y_hat.item(), pred[0].tolist()


def create_app():
    app = Flask(__name__, static_url_path="", static_folder="../web")

    @app.route("/")
    def root():
        return current_app.send_static_file('index.html')

    @app.route("/genres", methods=["GET"])
    def classes():
        return jsonify(class_labels)

    @app.route("/predict", methods=["POST"])
    def predict():
        file = request.files["file"]
        audio_bytes = file.read()
        data = create_input_from_bytes(audio_bytes)
        best_id, all_preds = inference(data)

        all_preds = {class_labels[i]: all_preds[i] for i in range(len(all_preds))}

        return jsonify({"class_id": best_id, "class_name": class_labels[best_id], "all_preds": all_preds})

    return app


async def main(args):
    app = create_app()
    app.run(debug=True, use_reloader=True)

