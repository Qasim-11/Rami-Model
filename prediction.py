######## IMPORTS ##########
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
from tensorflow.keras.models import load_model

####### ALL CONSTANTS #####
fs = 44100
seconds = 2
filename = "prediction.wav"
class_names = ["Wake Word NOT Detected", "Wake Word Detected"]

##### LOADING OUR SAVED MODEL and PREDICTING ###
model = load_model("D:\\Coding\\Project-2\\speech\\WakeWordDetection\\saved_model\\WWD_waded2.h5")

audio_file = "D:\\Coding\\Project-2\\speech\\WakeWordDetection\\videoplayback.wav"
voice, sample_rate = librosa.load(audio_file, sr=44100)  # Load with 44.1kHz sample rate

print("Prediction Started: ")
i = 0
while True:
    print("Say Now: ")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()
    write(filename, fs, myrecording)

    audio, sample_rate = librosa.load(filename)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfcc_processed = np.mean(mfcc.T, axis=0)

    prediction = model.predict(np.expand_dims(mfcc_processed, axis=0))
    if prediction[:, 1] > 0.99:
        print(f"Wake Word Detected for ({i})")
        print("Confidence:", prediction[:, 1])
        i += 1
        sd.play(voice, sample_rate)
        sd.wait()  
    else:
        print(f"Wake Word NOT Detected")
        print("Confidence:", prediction[:, 0])
