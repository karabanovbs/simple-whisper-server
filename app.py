from flask import Flask, request
import os
import whisper
import requests
from urllib.parse import unquote
import uuid


modelTiny = whisper.load_model("tiny")
modelTurbo = whisper.load_model("turbo")
app = Flask(__name__)


@app.route("/transcribe")
def transcribe():
    url = request.args.get('url', '')

    url = unquote(url)

    model = request.args.get('model', '')

    path = download(url, dest_folder='tmp')


    target_model = modelTurbo
    if model == 'tiny':
        target_model = modelTiny
    if model == 'turbo':
        target_model = modelTurbo

    result = target_model.transcribe(path)
    os.remove(path)

    return result["text"]


def download(url: str, dest_folder: str):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)  # create folder if it does not exist

    origin_file_name = url.split('/')[-1].replace(" ", "_")

    filename, file_extension = os.path.splitext(origin_file_name)

    # be careful with file names
    file_path = os.path.join(dest_folder, origin_file_name.replace(filename, str(uuid.uuid4())))

    r = requests.get(url, stream=True)
    if r.ok:
        print("saving to", os.path.abspath(file_path))
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
    else:  # HTTP status code 4XX/5XX
        print("Download failed: status code {}\n{}".format(r.status_code, r.text))
    return os.path.abspath(file_path)


app.run(host='0.0.0.0', port=8080)
