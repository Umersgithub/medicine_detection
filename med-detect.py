import os
import cv2
import numpy as np
from gtts import gTTS
from playsound import playsound
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Define constants
MODEL_PATH = 'medicine_model.h5'
CLASS_NAMES_PATH = 'class_names.txt'
INPUT_SIZE = (224, 224)  # Adjust based on your model's input size

def load_class_names(file_path):
    with open(file_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names

def preprocess_image(image):
    image = cv2.resize(image, INPUT_SIZE)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize pixel values
    return image

def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save("output.mp3")
    playsound("output.mp3")
    os.remove("output.mp3")

def main():
    # Load the trained model
    model = load_model(MODEL_PATH)
    
    # Load class names
    class_names = load_class_names(CLASS_NAMES_PATH)
    
    # Initialize the camera
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess the frame
        processed_frame = preprocess_image(frame)
        
        # Make prediction
        predictions = model.predict(processed_frame)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        # Display result on frame
        label = f"{class_names[predicted_class]}: {confidence:.2f}"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Speak out the detected medicine name if confidence is above a threshold
        if confidence > 0.8:  # You can adjust this threshold based on your needs
            speak(f"This is {class_names[predicted_class]}")
        
        # Show the frame
        cv2.imshow('Medicine Detection', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
