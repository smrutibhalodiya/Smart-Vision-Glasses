
import numpy as np
import cv2
import tkinter as tk
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from PIL import Image, ImageTk

class Classifier:
    def __init__(self):
        self.window = tk.Tk()
        self.cap1 = None
        self.Load_Model_Arch()
        self.load_trained_model()
        self.init_gui()

    def Load_Model_Arch(self):
        self.classifier = Sequential()
        self.classifier.add(Conv2D(32, (2, 2), input_shape=(160, 80, 3), activation='relu'))
        self.classifier.add(MaxPooling2D(pool_size=(2, 2)))
        self.classifier.add(Conv2D(64, (2, 2), activation='relu'))
        self.classifier.add(MaxPooling2D(pool_size=(2, 2)))
        self.classifier.add(Conv2D(128, (2, 2), activation='relu'))
        self.classifier.add(MaxPooling2D(pool_size=(2, 2)))
        self.classifier.add(Conv2D(128, (2, 2), activation='relu'))
        self.classifier.add(MaxPooling2D(pool_size=(2, 2)))
        self.classifier.add(Flatten())
        self.classifier.add(Dense(units=64, activation='relu'))
        self.classifier.add(Dense(units=1, activation='sigmoid'))
        self.classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print("\n[INFO] Architecture Loaded...")

    def load_trained_model(self):
        self.classifier.load_weights('Trained_model.keras')
        print("\n[INFO] Model Loaded Successfully...")

    def detect(self):
        print("\n[INFO] Camera Starting")
        print("\n[INFO] Press 'q' to exit")
        self.cap1 = cv2.VideoCapture(0)
        self.update_frame()

    def update_frame(self):
        ret1, frame1 = self.cap1.read()
        if not ret1:
            print("[ERROR] Failed to capture image")
            self.cap1.release()
            return

        frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        img = np.array(frame)
        bboxes = []
        for i in range(8):
            crop_img = img[:, 80 * i:160 + 80 * i]
            crop = cv2.resize(crop_img, (80, 160))
            test_image = np.expand_dims(crop, axis=0)
            result = self.classifier.predict(test_image)
            if result[0][0] > 0.5:
                bboxes.append([80 * i, 10, 80 * i + 160, 460])

        if bboxes:
            bboxes = np.array(bboxes)
            final_bboxes = self.non_max_suppression_fast(bboxes, 0.3)
            for (startX, startY, endX, endY) in final_bboxes:
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 210, 0), 3)

            cv2.putText(frame, "Person Ahead", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        im_pil = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=im_pil)
        self.img_label.imgtk = imgtk
        self.img_label.config(image=imgtk)

        self.window.after(10, self.update_frame)

    def non_max_suppression_fast(self, boxes, overlapThresh):
        if len(boxes) == 0:
            return []
        
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")
        
        pick = []
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)
        
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            
            overlap = (w * h) / area[idxs[:last]]
            
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
        
        return boxes[pick].astype("int")

    def init_gui(self):
        self.window.title("Pedestrian Detection")
        self.window.geometry("800x600")
        self.btn_detect = tk.Button(self.window, text="Detect Pedestrian", width=50, command=self.detect)
        self.btn_detect.pack(anchor=tk.CENTER, expand=True)
        self.img_label = tk.Label(self.window)
        self.img_label.pack(anchor=tk.CENTER, expand=True)
        
        self.window.bind('<q>', self.quit)
        self.window.mainloop()

    def quit(self, event=None):
        if self.cap1:
            self.cap1.release()
        self.window.destroy()

Classifier()
