import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk
import pyttsx3


class SignLanguagConvertor:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Translator")

        # Load the trained model and class names
        self.model = self.load_trained_model("sign_language_model.h5")
        self.class_names = self.load_class_names("class_names.txt")

        # MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.running = False
        self.speaking = False

        # Text-to-Speech Engine
        self.tts_engine = pyttsx3.init()

        # GUI elements
        self.camera_frame = tk.Canvas(root, width=640, height=480)
        self.camera_frame.grid(row=0, column=0, padx=10, pady=10)

        self.label_var = tk.StringVar()
        self.label_var.set("Prediction: None")
        self.label_display = tk.Label(root, textvariable=self.label_var, font=("Arial", 16), fg="blue")
        self.label_display.grid(row=1, column=0)

        self.start_button = tk.Button(root, text="Start", command=self.toggle_running, width=20, height=2)
        self.start_button.grid(row=2, column=0, pady=10)

        self.speak_button = tk.Button(root, text="Speak", command=self.toggle_speaking, width=20, height=2)
        self.speak_button.grid(row=3, column=0, pady=10)

        self.exit_button = tk.Button(root, text="Exit", command=self.exit_app, width=20, height=2)
        self.exit_button.grid(row=4, column=0, pady=10)

        # Continuously update the camera feed
        self.update_frame()

    def load_trained_model(self, model_path):
        """Loads the pre-trained model."""
        return load_model(model_path)

    def load_class_names(self, class_names_path):
        """Loads class names from a file."""
        with open(class_names_path, "r") as f:
            return [line.strip() for line in f]

    def detect_hand_landmarks(self, image):
        """Detects hand landmarks using MediaPipe."""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.hands.process(image_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
                )
            return image, result.multi_hand_landmarks
        return image, None

    def preprocess_hand_region(self, image, hand_landmarks, img_size=(128, 128)):
        """Preprocesses the hand region for prediction."""
        h, w, _ = image.shape
        x_min = int(min([lm.x for lm in hand_landmarks]) * w)
        x_max = int(max([lm.x for lm in hand_landmarks]) * w)
        y_min = int(min([lm.y for lm in hand_landmarks]) * h)
        y_max = int(max([lm.y for lm in hand_landmarks]) * h)

        hand_img = image[max(0, y_min):min(h, y_max), max(0, x_min):min(w, x_max)]
        hand_img = cv2.resize(hand_img, img_size)
        hand_img = hand_img / 255.0  # Normalize
        return np.expand_dims(hand_img, axis=0)

    def toggle_running(self):
        """Toggles the running state of the app."""
        self.running = not self.running
        self.start_button.config(text="Stop" if self.running else "Start")

    def toggle_speaking(self):
        """Toggles the speaking state of the app."""
        self.speaking = not self.speaking
        self.speak_button.config(text="Stop Speaking" if self.speaking else "Speak")

    def update_frame(self):
        """Updates the camera feed in the GUI."""
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            annotated_image, hand_landmarks_list = self.detect_hand_landmarks(frame)

            if self.running and hand_landmarks_list:
                for hand_landmarks in hand_landmarks_list:
                    try:
                        preprocessed_image = self.preprocess_hand_region(frame, hand_landmarks.landmark)
                        prediction = self.model.predict(preprocessed_image)
                        predicted_class = self.class_names[np.argmax(prediction)]
                        self.label_var.set(f"Prediction: {predicted_class}")

                        # Speak the prediction if enabled
                        if self.speaking and predicted_class != "None":
                            self.tts_engine.say(predicted_class)
                            self.tts_engine.runAndWait()
                    except Exception as e:
                        print(f"Error during prediction: {e}")
            else:
                self.label_var.set("Prediction: None")

            # Update the GUI with the camera feed
            img = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_frame.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.camera_frame.imgtk = imgtk

        self.root.after(10, self.update_frame)

    def exit_app(self):
        """Exits the application and releases resources."""
        self.cap.release()
        self.hands.close()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguagConvertor(root)
    root.mainloop()
