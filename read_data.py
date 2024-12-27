import os
import cv2
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk
import mediapipe as mp


class SignDataCollector:
    def __init__(self, root, data_dir="sign_data"):
        self.root = root
        self.root.title("Collect data")
        self.data_dir = data_dir
        self.img_size = (640, 480)

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2)
        self.mp_drawing = mp.solutions.drawing_utils
        self.capture = cv2.VideoCapture(0)
        self.running = False
        self.image_count = 0
        self.current_label = None

        # Create data directory if it doesn't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        # Set up the GUI
        self.canvas = tk.Canvas(root, width=self.img_size[0], height=self.img_size[1])
        self.canvas.pack()

        self.start_button = tk.Button(root, text="Start", command=self.start_collection, width=20, height=2)
        self.start_button.pack(pady=10)

        self.stop_button = tk.Button(root, text="Stop", command=self.stop_collection, width=20, height=2, state="disabled")
        self.stop_button.pack(pady=10)

        self.exit_button = tk.Button(root, text="Exit", command=self.exit_app, width=20, height=2)
        self.exit_button.pack(pady=10)

        # Continuously update the camera feed
        self.update_frame()

    def start_collection(self):
        # Get the label name
        label = simpledialog.askstring("Input", "Enter the label name:")
        if not label:
            messagebox.showwarning("Warning", "Label name cannot be empty!")
            return

        self.current_label = label
        label_dir = os.path.join(self.data_dir, label)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        else:
            messagebox.showerror("Duplicate error",f"The data for {label} already exist")
            return

        self.image_count = 0
        self.running = True
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        messagebox.showinfo("Info", f"Started capturing data for label: {label}")

    def stop_collection(self):
        self.running = False
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        messagebox.showinfo("Info", f"Captured {self.image_count} images for label: {self.current_label}")

    def update_frame(self):
        ret, frame = self.capture.read()
        if ret:
            # Flip the frame for mirror effect
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect hands using MediaPipe
            results = self.hands.process(rgb_frame)

            # Draw landmarks if hands are detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                # Save the processed frame
                if self.running and self.image_count < 50:
                    label_dir = os.path.join(self.data_dir, self.current_label)
                    file_path = os.path.join(label_dir, f"{self.image_count}.jpg")
                    cv2.imwrite(file_path, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    self.image_count += 1

                # Automatically stop after 50 images
                if self.image_count >= 50:
                    self.stop_collection()

            # Display label and count on the frame
            if self.running:
                cv2.putText(frame, f"Label: {self.current_label} | Count: {self.image_count}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Convert the frame for Tkinter
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.imgtk = imgtk

        # Keep updating the frame
        self.root.after(10, self.update_frame)

    def exit_app(self):
        self.capture.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = SignDataCollector(root)
    root.mainloop()
