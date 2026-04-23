import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import os
from datetime import datetime

# =============================
# Modern Professional Light UI Theme
# =============================
PRIMARY_BG = "#f8fafc"      # light background
HEADER_BG = "#2563eb"       # professional blue
CARD_BG = "#ffffff"         # white card
BUTTON_COLOR = "#2563eb"
SUCCESS_COLOR = "#16a34a"
TEXT_COLOR = "#0f172a"

# =============================
# Setup
# =============================

save_folder = "detected_faces"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = None
running = False
last_frame = None

# =============================
# Functions
# =============================

def start_camera():
    global cap, running

    if running:
        return

    cap = cv2.VideoCapture(0)
    running = True
    status_var.set("Camera Running")

    update_frame()


def stop_camera():
    global cap, running

    running = False

    if cap is not None:
        cap.release()

    status_var.set("Camera Stopped")


def capture_image():
    global last_frame

    if last_frame is None:
        messagebox.showerror("Error", "No frame available!")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{save_folder}/face_{timestamp}.jpg"

    cv2.imwrite(filename, last_frame)

    messagebox.showinfo("Success", "Image saved successfully!")


def update_frame():
    global cap, running, last_frame

    if not running:
        return

    ret, frame = cap.read()

    if ret:
        last_frame = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=6,
            minSize=(50, 50)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (22, 163, 74),
                2
            )

            cv2.putText(
                frame,
                "Face Detected",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (22, 163, 74),
                2
            )

        face_count_var.set(f"Faces Detected: {len(faces)}")

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        img = img.resize((700, 500))

        imgtk = ImageTk.PhotoImage(image=img)

        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        video_label.after(10, update_frame)

# =============================
# UI Layout (Clean Professional)
# =============================

root = tk.Tk()
root.title("AI Face Detection System")
root.geometry("1100x750")
root.configure(bg=PRIMARY_BG)

# Header

header_frame = tk.Frame(root, bg=HEADER_BG, height=80)
header_frame.pack(fill="x")

header_label = tk.Label(
    header_frame,
    text="AI FACE DETECTION SYSTEM",
    font=("Segoe UI", 26, "bold"),
    bg=HEADER_BG,
    fg="white"
)

header_label.pack(pady=18)

# Main Section

main_frame = tk.Frame(root, bg=PRIMARY_BG)
main_frame.pack(pady=20)

# Video Card

video_card = tk.Frame(
    main_frame,
    bg=CARD_BG,
    bd=2,
    relief="groove"
)

video_card.grid(row=0, column=0, padx=20)

video_label = tk.Label(video_card, bg="black")
video_label.pack(padx=15, pady=15)

# Right Panel

side_panel = tk.Frame(
    main_frame,
    bg=CARD_BG,
    bd=2,
    relief="groove",
    width=260,
    height=300
)

side_panel.grid(row=0, column=1, padx=15, sticky="n")

panel_title = tk.Label(
    side_panel,
    text="System Information",
    font=("Segoe UI", 16, "bold"),
    bg=CARD_BG,
    fg=TEXT_COLOR
)

panel_title.pack(pady=20)

face_count_var = tk.StringVar()
face_count_var.set("Faces Detected: 0")

face_label = tk.Label(
    side_panel,
    textvariable=face_count_var,
    font=("Segoe UI", 14, "bold"),
    bg=CARD_BG,
    fg=SUCCESS_COLOR
)

face_label.pack(pady=10)

status_var = tk.StringVar()
status_var.set("Camera Off")

status_label = tk.Label(
    side_panel,
    textvariable=status_var,
    font=("Segoe UI", 12),
    bg=CARD_BG,
    fg=TEXT_COLOR
)

status_label.pack(pady=10)

# Buttons Section

button_frame = tk.Frame(root, bg=PRIMARY_BG)
button_frame.pack(pady=25)

start_btn = tk.Button(
    button_frame,
    text="Start Camera",
    font=("Segoe UI", 12, "bold"),
    bg=BUTTON_COLOR,
    fg="white",
    width=15,
    command=start_camera
)

start_btn.grid(row=0, column=0, padx=15)

stop_btn = tk.Button(
    button_frame,
    text="Stop Camera",
    font=("Segoe UI", 12, "bold"),
    bg="#ef4444",
    fg="white",
    width=15,
    command=stop_camera
)

stop_btn.grid(row=0, column=1, padx=15)

capture_btn = tk.Button(
    button_frame,
    text="Capture Image",
    font=("Segoe UI", 12, "bold"),
    bg="#16a34a",
    fg="white",
    width=15,
    command=capture_image
)

capture_btn.grid(row=0, column=2, padx=15)

exit_btn = tk.Button(
    button_frame,
    text="Exit",
    font=("Segoe UI", 12, "bold"),
    bg="#0f172a",
    fg="white",
    width=15,
    command=root.destroy
)

exit_btn.grid(row=0, column=3, padx=15)

# Footer

footer = tk.Label(
    root,
    text="Developed by Deepika Gautam | Face Detection Project",
    font=("Segoe UI", 10),
    bg=PRIMARY_BG,
    fg="#475569"
)

footer.pack(side="bottom", pady=12)

root.mainloop()