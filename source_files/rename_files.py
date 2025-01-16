import os
import re

def clean_filename(filename):
    # Remove special characters (e.g., quotes) and replace spaces with underscores
    cleaned = re.sub(r'[^\w]', '_', filename)  # Keeps alphanumeric characters, underscores
    cleaned = re.sub(r'__+', '_', cleaned)  # Replaces multiple underscores with a single underscore
    #return cleaned.strip('_')  # Removes leading/trailing underscores if any
    return cleaned

def rename_wav_files (folder_path):
    for wav_file in os.listdir(folder_path):

        if '_wav_wav' in wav_file:  # Check if the filename contains "_wav_wav"
            # Create new filename by replacing "_wav_wav" with ".wav"
            new_filename = wav_file.replace('_wav_wav', '.wav')
            # Construct the full file paths
            old_filepath = os.path.join(folder_path, wav_file)
            new_filepath = os.path.join(folder_path, new_filename)
            # Rename the file
            os.rename(old_filepath, new_filepath)
        
        if wav_file.endswith(".wav"):
            original_path = os.path.join(folder_path, wav_file)

            #extract first few words (3) 
            base_name = '_'.join(wav_file.split()[:3])
            new_filename = clean_filename(base_name)
            #new_filename = f"{base_name}.wav"

            new_path = os.path.join(folder_path, new_filename)

            #Rename the file
            os.rename(original_path, new_path)
            print(f"Renamed: {original_path} to {new_path}")
        else:
            original_path = os.path.join(folder_path, wav_file)
            new_filename = f"{wav_file}.wav"
            new_path = os.path.join(folder_path, new_filename)
            #Rename the file
            os.rename(original_path, new_path)
            print(f"Renamed: {original_path} to {new_path}")

wav_folder = '/Users/geetapuri/phd/SUTD/2024/term3/MLproject_Dorien/podcasts/WeStudyBillionaires/waveFiles'

rename_wav_files(wav_folder)