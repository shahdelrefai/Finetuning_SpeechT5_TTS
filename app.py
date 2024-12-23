from flask import Flask, request, jsonify, send_from_directory, render_template
from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor
from huggingface_hub import login
from datasets import load_dataset, Dataset
import os
import torch
from speechbrain.inference import EncoderClassifier
from transformers import SpeechT5HifiGan
import soundfile as sf

api_token = "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"
login(token=api_token)

model = SpeechT5ForTextToSpeech.from_pretrained(
    "shahdelrefai/speecht5_finetuned_voxpopuli_en"
)
checkpoint = "microsoft/speecht5_tts"
processor = SpeechT5Processor.from_pretrained(checkpoint)

dataset = load_dataset("facebook/voxpopuli", "en", streaming=True, trust_remote_code=True)
test_dataset = dataset['test']

spk_model_name = "speechbrain/spkrec-xvect-voxceleb"

device = "cpu"
speaker_model = EncoderClassifier.from_hparams(
    source=spk_model_name,
    run_opts={"device": device},
    savedir=os.path.join("/tmp", spk_model_name),
)

def create_speaker_embedding(waveform):
    with torch.no_grad():
        speaker_embeddings = speaker_model.encode_batch(torch.tensor(waveform))
        speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
        speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()
    return speaker_embeddings

def prepare_dataset(example):
    audio = example["audio"]

    example = processor(
        text=example["normalized_text"],
        audio_target=audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_attention_mask=False,
    )

    # strip off the batch dimension
    example["labels"] = example["labels"][0]

    # use SpeechBrain to obtain x-vector
    example["speaker_embeddings"] = create_speaker_embedding(audio["array"])

    return example

test_dataset = test_dataset.map(prepare_dataset)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit_text():
    speaker_id = int(request.form.get('speaker_id'))
    example = third_item = next(x for i, x in enumerate(test_dataset) if i == speaker_id)
    speaker_embeddings = torch.tensor(example["speaker_embeddings"]).unsqueeze(0)
    text = request.form.get('text')
    inputs = processor(text=text, return_tensors="pt")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    sf.write('static/audio/audio.wav', speech.numpy(), 16000)

    return jsonify({"audio_url": "static/audio/audio.wav"})

if __name__ == '__main__':
    app.run(debug=True)