import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import os

# Load the processor and model
processor = WhisperProcessor.from_pretrained("openai/whisper-small.en")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small.en")

def segment_audio(file_path, segment_length=60):
    """Split the audio into segments of given length (in seconds)."""
    waveform, sample_rate = torchaudio.load(file_path)

    # Resample if not at 16 kHz
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)

    # Calculate the number of samples per segment
    samples_per_segment = segment_length * sample_rate
    num_segments = waveform.size(1) // samples_per_segment + (1 if waveform.size(1) % samples_per_segment != 0 else 0)

    segments = []
    for i in range(num_segments):
        start_sample = i * samples_per_segment
        end_sample = min((i + 1) * samples_per_segment, waveform.size(1))
        segments.append(waveform[:, start_sample:end_sample])

    return segments

def transcribe_segment(segment):
    """Transcribe a single audio segment."""
    # Ensure segment is 1D for Whisper
    segment = segment.squeeze()

    # Process audio for Whisper
    inputs = processor(segment, sampling_rate=16000, return_tensors="pt")

    # Generate transcription with extended max tokens for longer transcriptions
    with torch.no_grad():
        generated_ids = model.generate(inputs["input_features"], max_new_tokens=400)
    
    # Decode transcription
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription

def transcribe_audio_in_segments(file_path, segment_length=60):
    """Transcribe the audio file in chunks and combine the results."""
    segments = segment_audio(file_path, segment_length)
    full_transcription = []

    # Transcribe each segment and append to the final transcription
    for i, segment in enumerate(segments):
        print(f"Transcribing segment {i + 1}/{len(segments)}")
        transcription = transcribe_segment(segment)
        full_transcription.append(transcription)
    
    # Combine all transcriptions
    return " ".join(full_transcription)

def transcribe_all_files(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process each .wav file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            file_path = os.path.join(input_folder, filename)
            print(f"Transcribing {filename}...")

            # Transcribe the audio file
            transcription = transcribe_audio_in_segments(file_path)

            # Save the transcription to a text file
            output_file_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.txt")
            with open(output_file_path, "w") as text_file:
                text_file.write(transcription)
            
            print(f"Saved transcription for {filename} to {output_file_path}")

# Example usage
#audio_path = r'/Users/geetapuri/phd/SUTD/2024/term3/MLproject_Dorien/wavFiles/Apple ＂Scary Fast＂ and Qualcomm Snapdragon Summit Events [lP0KntsKTrM].wav'  
#transcription = transcribe_audio_in_segments(audio_path, segment_length=60)
#print("Full Transcription:", transcription)

# Define your input and output folders
input_folder = "/geeta/SentAnPod/podcasts/Pomp_Podcast/waveFiles"  # Replace with the path to your folder of .wav files
output_folder = "/geeta/SentAnPod/podcasts/Pomp_Podcast/txtFiles"  # Replace with the folder where transcriptions will be saved

# Run batch transcription
transcribe_all_files(input_folder, output_folder)

#import torchaudio

"""
file_path = "/geeta/SentAnPod/podcasts/Pomp_Podcast/waveFiles/245.wav"

try:
    waveform, sample_rate = torchaudio.load(file_path)
    print(f"Loaded file successfully! Waveform shape: {waveform.shape}, Sample rate: {sample_rate}")
except RuntimeError as e:
    print(f"Failed to load audio file: {e}")

"""
#waveform, sample_rate = torchaudio.load("/geeta/SentAnPod/podcasts/Pomp_Podcast/waveFiles/444.wav")
#print(torchaudio.list_audio_backends())
#print(torchaudio.__version__)
#print(torch.__version__)