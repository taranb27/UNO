import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
import torch
from torchvision import transforms
from model import NeuralNet  # Import your model class

# Initialize the model
model = NeuralNet()
model.load_state_dict(torch.load('UNO_CNN.pth'))
model.eval()

class_names = [
    '0_blue', '1_blue', '2_blue', '3_blue', '4_blue', '5_blue', '6_blue', '7_blue', '8_blue', '9_blue',
    '0_green', '1_green', '2_green', '3_green', '4_green', '5_green', '6_green', '7_green', '8_green', '9_green',
    '0_red', '1_red', '2_red', '3_red', '4_red', '5_red', '6_red', '7_red', '8_red', '9_red',
    '0_yellow', '1_yellow', '2_yellow', '3_yellow', '4_yellow', '5_yellow', '6_yellow', '7_yellow', '8_yellow', '9_yellow',
    'skip_blue', 'reverse_blue', 'draw2_blue',
    'skip_green', 'reverse_green', 'draw2_green',
    'skip_red', 'reverse_red', 'draw2_red',
    'skip_yellow', 'reverse_yellow', 'draw2_yellow',
    'wild', 'wild_draw4'
]

# Define the preprocessing steps
preprocess = transforms.Compose([
    transforms.Resize((585, 410)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def predict_card_from_image(image):
    input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(input_tensor)
    _, predicted_idx = torch.max(output, 1)  # Get the index of the highest score
    predicted_class = class_names[predicted_idx.item()]  # Map index to class name
    return predicted_class

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")

class UNOCardDetectorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("UNO Card Detector")
        self.geometry("800x600")
        self.sidebar_frame = ctk.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        
        self.label = ctk.CTkLabel(self.sidebar_frame, text="Options", font=("Arial", 18))
        self.label.grid(row=0, column=0, padx=20, pady=20)

        self.file_button = ctk.CTkButton(self.sidebar_frame, text="File Input", command=self.select_file)
        self.file_button.grid(row=1, column=0, padx=20, pady=10)

        self.camera_button = ctk.CTkButton(self.sidebar_frame, text="Camera Input", command=self.start_camera)
        self.camera_button.grid(row=2, column=0, padx=20, pady=10)

        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)

        self.main_label = ctk.CTkLabel(self.main_frame, text="Select an option to start UNO card detection.", font=("Arial", 14))
        self.main_label.pack(pady=20, padx=20)

        self.video_label = ctk.CTkLabel(self.main_frame, text="")
        self.video_label.pack()

        self.image_label = ctk.CTkLabel(self.main_frame, text="")
        self.image_label.pack(pady=20)

        self.uno_output_label = ctk.CTkLabel(self.main_frame, text="Detected UNO card : ", font=("Arial", 12), wraplength=600)
        self.uno_output_label.pack(side="bottom", pady=30)

        self.cap = None
        self.is_camera_running = False

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

    def select_file(self):
        self.stop_camera()
        
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self.main_label.configure(text=f"File selected: {os.path.basename(file_path)}\nProcessing...")
            self.detect_from_file(file_path)
        else:
            self.main_label.configure(text="No file selected.")
            messagebox.showwarning("No file selected", "Please select a file to proceed.")

    def start_camera(self):
        self.stop_camera()
        self.image_label.configure(image="")  
        self.main_label.configure(text="Starting camera...")

        if not self.is_camera_running:
            self.is_camera_running = True
            self.cap = cv2.VideoCapture(0)  
            self.update_camera_frame()

    def stop_camera(self):
        self.is_camera_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_label.configure(image="")  

    def update_camera_frame(self):
        # Check if camera is running
        if self.is_camera_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Convert the frame to RGB before showing in Tkinter
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk  
                self.video_label.configure(image=imgtk)
                
                # Predict the card from the frame
                pil_image = Image.fromarray(frame_rgb)
                predicted_class = predict_card_from_image(pil_image)
                self.uno_output_label.configure(text=f"Detected UNO card: {predicted_class}")
            self.after(10, self.update_camera_frame)
        else:
            self.stop_camera()

    def detect_from_file(self, file_path):
        # Load the image and display it
        try:
            image = Image.open(file_path)
            image = image.resize((400, 300), Image.LANCZOS) 
            imgtk = ImageTk.PhotoImage(image) 
            self.image_label.imgtk = imgtk  
            self.image_label.configure(image=imgtk)
            self.uno_output_label.configure(text=f"Detected UNO card : {file_path}")  
        except Exception as e:
            self.main_label.configure(text="Error loading image.")
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    app = UNOCardDetectorApp()
    app.mainloop()
