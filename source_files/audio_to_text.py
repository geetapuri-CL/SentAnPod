from pydub import AudioSegment
import speech_recognition as sr
import os
import time

def transcribe_mp3_batch(mp3_folder, output_folder, batch_size=10, batch_delay=10, retries=3):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    recognizer = sr.Recognizer()
    mp3_files = [f for f in os.listdir(mp3_folder) if f.endswith(".mp3")]
    
    # Process files in batches
    for i in range(0, len(mp3_files), batch_size):
        batch_files = mp3_files[i:i + batch_size]
        
        print(f"Processing batch {i // batch_size + 1} with {len(batch_files)} files...")
        
        for mp3_file in batch_files:
            audio_path = os.path.join(mp3_folder, mp3_file)
            try:
                # Convert MP3 to WAV in memory
                audio = AudioSegment.from_mp3(audio_path)
                temp_wav_path = "temp.wav"
                audio.export(temp_wav_path, format="wav")
                
                with sr.AudioFile(temp_wav_path) as source:
                    audio_data = recognizer.record(source)

                    # Retry logic for each file
                    for attempt in range(retries):
                        try:
                            # Use Google STT for transcription
                            text = recognizer.recognize_google(audio_data)
                            print(f"Transcribed {mp3_file}")

                            # Save transcription to text file
                            output_file_path = os.path.join(output_folder, f"{os.path.splitext(mp3_file)[0]}.txt")
                            with open(output_file_path, "w") as text_file:
                                text_file.write(text)
                            break  # Exit retry loop on success

                        except sr.RequestError as e:
                            print(f"Could not request results for {mp3_file}; {e}")
                            if attempt < retries - 1:
                                print("Retrying...")
                                time.sleep(2)  # Wait before retrying
                            else:
                                print(f"Failed after {retries} attempts.")

                        except sr.UnknownValueError:
                            print(f"Could not understand audio in {mp3_file}")
                            break  # No point in retrying if the audio is unintelligible
                
                # Clean up the temporary WAV file
                if os.path.exists(temp_wav_path):
                    os.remove(temp_wav_path)

            except Exception as e:
                print(f"Error processing file {mp3_file}: {e}")
        
        # Pause after each batch
        print(f"Completed batch {i // batch_size + 1}. Waiting {batch_delay} seconds before the next batch.")
        time.sleep(batch_delay)

# Define your folders and batch settings

mp3_folder = "/Users/geetapuri/phd/SUTD/2024/term3/MLproject_Dorien/podcasts/Pomp_Podcast/Pomp_Podcast"
output_folder = "/Users/geetapuri/phd/SUTD/2024/term3/MLproject_Dorien/txtFiles"
batch_size = 10         # Number of files per batch
batch_delay = 15        # Delay (in seconds) between batches
retries = 3             # Number of retry attempts per file

# Run the batch transcription
transcribe_mp3_batch(mp3_folder, output_folder, batch_size, batch_delay, retries)
