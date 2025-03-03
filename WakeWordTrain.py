import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential, layers
from tensorflow.keras.callbacks import EarlyStopping

##### Loading saved csv ##############
df = pd.read_pickle("final_audio_data_csv/audio_data.csv")

####### Making our data training-ready
X = df["feature"].values
X = np.concatenate(X, axis=0).reshape(len(X), 40)  # Assuming 40 MFCC features
X = np.expand_dims(X, axis=-1)  # Add channel dimension for Conv1D
y = np.array(df["class_label"].tolist())
y = to_categorical(y)

####### train test split ############
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.05, random_state=42)

##### GRU-Based Model Architecture ############
model = Sequential([
    layers.Input(shape=(40, 1)),  # Input: 40 MFCC features
    layers.Conv1D(16, kernel_size=3, activation='relu', padding='same'),  # Lightweight CNN for feature extraction
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPooling1D(pool_size=2),
    layers.GRU(64, return_sequences=True),  # First GRU layer
    layers.GRU(32),  # Second GRU layer
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),  # Regularization
    layers.Dense(2, activation='softmax')  # Binary classification
])

##### Compile the Model ############
model.compile(
    loss="categorical_crossentropy",
    optimizer='adam',
    metrics=['accuracy']
)


##### Train the Model ############
history = model.fit(
    X_train, y_train,
    epochs=1000,
    batch_size=32,
    validation_split=0.05
)

##### Save the Model ############
model.save("wake_word_gru_model.h5")

##### Evaluate the Model ############
score = model.evaluate(X_test, y_test)
print("Test Loss:", score[0])
print("Test Accuracy:", score[1])

##### Classification Report ############
from sklearn.metrics import classification_report, confusion_matrix

y_pred = np.argmax(model.predict(X_test), axis=1)
print("Classification Report:\n", classification_report(np.argmax(y_test, axis=1), y_pred))