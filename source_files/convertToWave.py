import os
import subprocess
from pydub import AudioSegment

def convert_mp3_to_wav(mp3_path, wav_path):
    for mp3_file in os.listdir(mp3_path):
        if mp3_file.endswith(".mp3"):
            
            # Extract the first 3 characters of the filename for the output
            wav_filename = mp3_file[:3] + ".wav"
            wav_file_path = os.path.join(wav_path, wav_filename)
            

            #full mp3 path
            full_mp3_path = os.path.join(mp3_path, mp3_file)

            audio = AudioSegment.from_mp3(full_mp3_path)
            audio = audio.set_frame_rate(16000).set_channels(1)  # Resample to 16 kHz
            audio.export(wav_file_path, format="wav")

            print("Wave file create at = ", wav_file_path)

def convert_opus_to_wav_with_ffmpeg(opus_folder, wav_folder):
    # Ensure the output folder exists
    os.makedirs(wav_folder, exist_ok=True)

    for opus_file in os.listdir(opus_folder):
        if opus_file.endswith(".opus"):
            opus_path = os.path.join(opus_folder, opus_file)
            wav_path = os.path.join(wav_folder, os.path.splitext(opus_file)[0] + ".wav")

            # Use FFmpeg to convert the file
            command = ["ffmpeg", "-y", "-i", opus_path, "-ar", "16000", "-ac", "1", wav_path]
            try:
                subprocess.run(command, check=True)
                print(f"Converted {opus_file} to {os.path.basename(wav_path)}")
            except subprocess.CalledProcessError as e:
                print(f"Error converting {opus_file}: {e}")


# Define your folders
#mp3_folder = "/Users/geetapuri/phd/SUTD/2024/term3/MLproject_Dorien/podcasts/Pomp_Podcast/Pomp_Podcast_mp3"
mp3_folder = "/geeta/SentAnPod/podcasts/Pomp_Podcast/Pomp_Podcast_mp3"
#output_folder = "/Users/geetapuri/phd/SUTD/2024/term3/MLproject_Dorien/txtFiles"

# Define your input folder with .opus files and output folder for .wav files
#opus_folder = '/Users/geetapuri/phd/SUTD/2024/term3/MLproject_Dorien/podcasts/TechCheck'

opus_folder = '/Users/geetapuri/phd/SUTD/2024/term3/MLproject_Dorien/podcasts/WeStudyBillionaires/opus'
#wav_folder = '/Users/geetapuri/phd/SUTD/2024/term3/MLproject_Dorien/podcasts/WeStudyBillionaires/waveFiles'
wav_folder = '/geeta/SentAnPod/podcasts/Pomp_Podcast/waveFiles'
# Run the conversion
#convert_opus_to_wav_with_ffmpeg(opus_folder, wav_folder)

convert_mp3_to_wav(mp3_folder, wav_folder)