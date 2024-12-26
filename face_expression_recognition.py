import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv('fer2013.csv')
X = []
y = []

for index, row in data.iterrows():
    # Convert string of pixel values to array
    pixels = np.array(row['pixels'].split(), dtype='float32')
    X.append(pixels.reshape(48, 48, 1))  # Reshape to 48x48 pixels
    y.append(row['emotion'])  # Labels

X = np.array(X)
y = np.array(y)

# Preprocess data
X /= 255.0  # Normalize pixel values
y = to_categorical(y, num_classes=7)  # Convert labels to one-hot encoding

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))  # 7 classes for emotions

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=64)

# Save the model
model.save('emotion_recognition_model.h5')
# Load the model
from tensorflow.keras.models import load_model

model = load_model('emotion_recognition_model.h5')

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


# Function for real-time detection
def recognize_expression():
    cap = cv2.VideoCapture(0)  # Start webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        detected_faces = faces.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in detected_faces:
            face = gray_frame[y:y + h, x:x + w]
            face_resized = cv2.resize(face, (48, 48))
            face_normalized = face_resized.astype('float32') / 255.0
            face_reshaped = np.reshape(face_normalized, (1, 48, 48, 1))

            prediction = model.predict(face_reshaped)
            emotion_index = np.argmax(prediction)
            emotion = emotion_labels[emotion_index]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        cv2.imshow('Real-time Face Expression Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Call the function to start detection
recognize_expression()
