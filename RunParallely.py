######## IMPORTS ##########
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import concurrent.futures
import queue
import time
import pyttsx3

#### SETTING UP TEXT TO SPEECH ###
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', 'HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_DAVID_11.0')

audio_file = "D:\\Coding\\Project-2\\speech\\WakeWordDetection\\videoplayback.wav"
voice, sample_rate = librosa.load(audio_file, sr=44100)  # Load with 44.1kHz sample rate



####### ALL CONSTANTS #####
fs = 44100  # Sample rate
window_size = 1.0  # Process 1-second audio chunks
overlap = 0.5  # Overlap of 0.5 seconds between chunks
filename = "prediction.wav"
class_names = ["Wake Word NOT Detected", "Wake Word Detected"]

##### LOADING OUR SAVED MODEL #####
model = load_model("wake_word_gru_model.h5")

# Queue to hold audio chunks for processing
audio_queue = queue.Queue()
wake_just_detected = False
wake_counter = 0
def record_audio():
    """Record audio continuously and split it into overlapping chunks."""
    print("Recording started...")
    chunk_size = int(window_size * fs)
    step_size = int((window_size - overlap) * fs)
    buffer = np.zeros((chunk_size, 2), dtype=np.float32)

    with sd.InputStream(samplerate=fs, channels=2, dtype='float32') as stream:
        while True:
            # Read audio data from the stream
            data, _ = stream.read(step_size)
            buffer = np.roll(buffer, -step_size, axis=0)
            buffer[-step_size:, :] = data

            # Add the chunk to the queue for processing
            audio_queue.put(buffer.copy())

def process_audio():
    """Process audio chunks from the queue and predict wake word."""
    print("Processing started...")
    while True:

        # if wake_just_detected:
        #     if time.time() - t > 2:
        #         wake_just_detected = False
        # Get the next audio chunk from the queue
        audio_chunk = audio_queue.get()

        # Save the chunk to a file (optional, for debugging)
        write(filename, fs, audio_chunk)

        # Extract MFCC features
        audio, sample_rate = librosa.load(filename, sr=fs)
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfcc_processed = np.mean(mfcc.T, axis=0)

        # Predict wake word
        prediction = model.predict(np.expand_dims(mfcc_processed, axis=0))
        if prediction[:, 1] > 0.9:
            print("Wake Word Detected!")
            print("Confidence:", prediction[:, 1])
            sd.play(voice, sample_rate)

                        
        else:
            print("Wake Word NOT Detected")
            print("Confidence:", prediction[:, 0])
        


def speak(audio):
    engine.say(audio)
    engine.runAndWait()

# Start recording and processing in parallel
with concurrent.futures.ThreadPoolExecutor() as executor:
    # Start the recording thread
    executor.submit(record_audio)

    # Start the processing thread
    executor.submit(process_audio)


