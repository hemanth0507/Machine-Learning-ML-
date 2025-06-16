from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2

# Constants
IMG_SIZE = 224
BATCH_SIZE = 32

# Prepare data generators
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    r'C:\Users\bdhan\Downloads\Face_Mask_Detection',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    r'C:\Users\bdhan\Downloads\Face_Mask_Detection',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# 🔍 Check how labels are mapped
print("Class indices:", train_generator.class_indices)
# e.g., Output: {'mask': 0, 'no_mask': 1}

# Build CNN model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile and train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(train_generator, epochs=5, validation_data=val_generator)

# Save the model
model.save('Real_World_Mask_Detection.h5')

# Load the model
model = keras.models.load_model('Real_World_Mask_Detection.h5')
print("Model Loaded")

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        try:
            face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            face_array = face_resized / 255.0
            face_array = np.expand_dims(face_array, axis=0)

            prediction = model.predict(face_array)[0][0]
            print("Prediction Score:", prediction)  # 🔍 Debugging

            # 🔁 Use correct logic based on class_indices
            # If {'mask': 0, 'no_mask': 1}, this is correct:
            label = "No Mask" if prediction < 0.5 else "Mask"

            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        except Exception as e:
            print("Error processing face:", e)
            continue

    cv2.imshow("Face Mask Detector", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
x*