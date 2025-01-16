import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torchaudio
import os

# Load the processor and model from Hugging Face
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

from pydub import AudioSegment

def convert_mp3_to_wav(mp3_path, wav_path):
    audio = AudioSegment.from_mp3(mp3_path)
    audio = audio.set_frame_rate(16000).set_channels(1)  # Resample to 16 kHz
    audio.export(wav_path, format="wav")


def transcribe_audio(file_path):
    # Load audio
    waveform, sample_rate = torchaudio.load(file_path)
    # Resample if necessary
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

    # Ensure the waveform has shape [batch_size, channels, sequence_length]
    #waveform = waveform.unsqueeze(0)  # Now [1, channels, sequence_length]
    waveform = waveform.unsqueeze(0).squeeze()
    
    # Process the audio
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)

    # Perform inference
    with torch.no_grad():
        logits = model(inputs.input_values).logits

    # Decode the logits to text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    return transcription



def transcribe_batch(mp3_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for mp3_file in os.listdir(mp3_folder):
        if mp3_file.endswith(".mp3"):
            #wav_file_path = os.path.join(output_folder, os.path.splitext(mp3_file)[0] + ".wav")
            wav_file_path = os.path.join(output_folder, os.path.splitext(mp3_file)[0][-4:] + ".wav")

            convert_mp3_to_wav(os.path.join(mp3_folder, mp3_file), wav_file_path)

            print("Wave file path = ", wav_file_path)

            transcription = transcribe_audio(wav_file_path)

            # Save transcription
            text_file_path = os.path.join(output_folder, os.path.splitext(mp3_file)[0] + ".txt")
            with open(text_file_path, "w") as f:
                f.write(transcription)
            print(f"Transcribed {mp3_file}")

# Define your folders
mp3_folder = "/Users/geetapuri/phd/SUTD/2024/term3/MLproject_Dorien/podcasts/Pomp_Podcast/Pomp_Podcast"
output_folder = "/Users/geetapuri/phd/SUTD/2024/term3/MLproject_Dorien/txtFiles"

# Run batch transcription
transcribe_batch(mp3_folder, output_folder)

