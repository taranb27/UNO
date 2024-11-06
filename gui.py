import torch
import torch.nn as nn
import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np

# Import the neuralnn class from the test.py file
from model import neuralnn  # Make sure test.py is in the same directory or adjust the import path

# Instantiate the model
model = neuralnn()

# Load the state dict from the saved model
model.load_state_dict(torch.load('UNO_CNN.pth'))
model.eval()  # Set the model to evaluation mode

# Preprocess the image
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, (640, 640))
    image_tensor = torch.from_numpy(image_resized).float()
    image_tensor = image_tensor.permute(2, 0, 1)  # Convert HWC to CHW format
    image_tensor /= 255.0
    return image_tensor.unsqueeze(0)  # Add batch dimension

# Main GUI class
class CardRecognitionApp:
    def __init__(self, window):
        self.window = window
        self.window.title("UNO Card Recognition")

        # Button to open webcam
        self.webcam_button = tk.Button(window, text="Open Webcam", command=self.open_webcam)
        self.webcam_button.pack()

        # Button to open file dialog
        self.file_button = tk.Button(window, text="Select File", command=self.select_file)
        self.file_button.pack()

        # Label to display recognized card
        self.result_label = tk.Label(window, text="Card Recognition Result", font=("Arial", 14))
        self.result_label.pack()

    def open_webcam(self):
        self.capture = cv2.VideoCapture(0)
        self.process_video()

    def select_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.process_image(file_path)

    def process_image(self, file_path):
        # Load the image from file
        image = cv2.imread(file_path)
        self.recognize_card(image)

    def process_video(self):
        # Capture frame-by-frame from webcam
        ret, frame = self.capture.read()
        if ret:
            self.recognize_card(frame)
            cv2.imshow('Webcam', frame)
            self.window.after(10, self.process_video)

    def recognize_card(self, image):
        # Preprocess the image
        image_tensor = preprocess_image(image)

        # Perform inference with the trained model
        with torch.no_grad():
            results = model(image_tensor)

        # Assuming you're using a classification model
        card_class = results.argmax().item()  # Get the predicted class (if classification)
        card_label = f"Recognized Card: {card_class}"  # Map this class to your card labels
        self.result_label.config(text=card_label)
