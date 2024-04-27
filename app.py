import tkinter as tk
import cv2 as cv
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from PIL import Image, ImageTk

class FacialMaskDetectionApp:
    def __init__(self, master):
        self.master = master
        master.title("Facial Mask Detection")

        self.label = tk.Label(master, text="Webcam Feed")
        self.label.pack()

        self.start_button = tk.Button(master, text="Start Detection", command=self.start_detection)
        self.start_button.pack()

        self.stop_button = tk.Button(master, text="Stop Detection", command=self.stop_detection, state=tk.DISABLED)
        self.stop_button.pack()

        self.video_feed_label = tk.Label(master)
        self.video_feed_label.pack()

        self.pretrained = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.capture = cv.VideoCapture(0)
        self.model = SVC()
        self.pca = PCA(n_components=3)

        # Load the mask and no mask data
        mask = np.load('mask.npy')
        no_mask = np.load('no_mask.npy')
        mask = mask.reshape(200, 50 * 50 * 3)
        no_mask = no_mask.reshape(200, 50 * 50 * 3)
        X = np.r_[no_mask, mask]
        y = np.zeros(X.shape[0])
        y[200:] = 1.0
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.20)
        self.X_train = self.pca.fit_transform(X_train)
        self.model.fit(self.X_train, y_train)

        self.detect_face = False

        self.update()

    def start_detection(self):
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.detect_face = True

    def stop_detection(self):
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.detect_face = False

    def update(self):
        ret, img = self.capture.read()
        if ret:
            if self.detect_face:
                self.detect_and_display_mask(img)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(image=img)
            self.video_feed_label.img = img
            self.video_feed_label.config(image=img)
        self.master.after(10, self.update)

    def detect_and_display_mask(self, img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = self.pretrained.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face_color = img[y:y + h, x:x + w]
            face = cv.resize(face, (50, 50))
            face = face.reshape(1, -1)
            # Ensure face has the correct number of features (7500)
            face = np.pad(face, ((0, 0), (0, 7500 - len(face[0]))), mode='constant')
            face = self.pca.transform(face)
            pred = self.model.predict(face)
            if int(pred) == 0:
                cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv.putText(img, 'No Mask', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            else:
                cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv.putText(img, 'Mask', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

def main():
    root = tk.Tk()
    app = FacialMaskDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
