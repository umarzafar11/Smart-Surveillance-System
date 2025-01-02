import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from keras.models import load_model
from PIL import Image
import shutil
import os
#from playsound import playsound
from alert import Ui_Dialogc
import sys
import numpy as np

import cv2
from PyQt5.QtWidgets import QVBoxLayout, QFileDialog

from PyQt5 import QtCore, QtGui, QtWidgets
#from PyQt5 import QtCore
from PyQt5.QtCore  import pyqtSlot
from PyQt5.QtGui import QImage , QPixmap
from PyQt5.QtWidgets import QDialog ,QMainWindow, QApplication
from PyQt5.uic import loadUi
from test1 import App
from PyQt5.QtCore import  QDir

model_path = r'G:\Umar Zafar Data\8th Semester\FYP\Fire Detection\Trained model\fire_model.h5'
model = tf.keras.models.load_model(model_path)


model_2 = load_model('E:\slowfast_finalmodel.hd5')

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(40, 170, 75, 51))
        self.pushButton.setObjectName("pushButton")
        self.pushButton2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton2.setGeometry(QtCore.QRect(70, 400, 75, 71))
        self.pushButton2.setObjectName("pushButton2")
        
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.pushButton.clicked.connect(self.camera_open)
        self.pushButton2.clicked.connect(self.open_dialog)
        
    def camera_open(self):
        video = cv2.VideoCapture(0)
        while True:
            _, frame = video.read()
            #Convert the captured frame into RGB
            im = Image.fromarray(frame, 'RGB')
            #Resizing into 224x224 because we trained the model with this image size.
            im = im.resize((224,224))
            img_array = image.img_to_array(im)
            img_array = np.expand_dims(img_array, axis=0) / 255
            probabilities = model.predict(img_array)[0]
            #Calling the predict method on model to predict 'fire' on the image
            prediction = np.argmax(probabilities)
            #if prediction is 0, which means there is fire in the frame.

            #if prediction == 0:
            if probabilities[prediction] >= 0.92:
                #label = 'Fire Detected'
                color = (0,255,0)
                font = cv2.FONT_HERSHEY_TRIPLEX 
  
                cv2.putText(frame, 
                'FIRE DETECTED', 
                (50, 50), 
                font, 1, 
                (0,0,255), 
                2, 
                cv2.LINE_4)
                #alert_box()
                #cv2.waitKey(1)

                
                #self.Dialog = QtWidgets.QDialog()
                #self.ui = Ui_Dialogc()
                #self.ui.setupUi(self.Dialog)
                #self.Dialog.show()

                
                #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                #cv2.putText(frame, label,
                   # cv2.FONT_HERSHEY_SIMPLEX, color,fontScale = 1, thickness =2)
                print(probabilities[prediction])
            cv2.imshow("Capturing", frame)
            #key=cv2.waitKey(1)
            key=cv2.waitKey(500)
            if key ==32:
                #cv2.waitkey()
                self.Dialog = QtWidgets.QDialog()
                self.ui = Ui_Dialogc()
                self.ui.setupUi(self.Dialog)
                self.Dialog.show()
            elif key == ord('q'):
                
                
                break
        video.release()
        cv2.destroyAllWindows()


    def alert_box(self):
        self.Dialog = QtWidgets.QDialog()
        self.ui = Ui_Dialogc()
        self.ui.setupUi(self.Dialog)
        self.Dialog.show()
      
    def open_file(self):
        try:
            path = QFileDialog.getOpenFileName(self, 'Open a file', '',
                                        'All Files (*.*)')
            if path != ('', ''):
                print("File path : "+ path[0])
                
        except Exception as e:
            print(e)
    
    def open_dialog(self):
        def frames_from_video(video_dir, nb_frames = 25, img_size = 224):
            

            # Opens the Video file
            cap = cv2.VideoCapture(video_dir)
            i=0
            frames = []
            while(cap.isOpened() and i<nb_frames):
                ret, frame = cap.read()
                if ret == False:
                    break
                frame = cv2.resize(frame, (img_size, img_size))
                frames.append(frame)
                i+=1

            cap.release()
            cv2.destroyAllWindows()
            return np.array(frames) / 255.0




        def predictions(video_dir, model, nb_frames = 25, img_size = 224):

            X = frames_from_video(video_dir, nb_frames, img_size)
            X = np.reshape(X, (1, nb_frames, img_size, img_size, 3))
    
            predictions = model.predict(X)
            preds = predictions.argmax(axis = 1)

            classes = []
            with open(os.path.join('F:\Test Videos\classes.txt'), 'r') as fp:
                for line in fp:
                    classes.append(line.split()[1])

            for i in range(len(preds)):
                print('Prediction - {} -- {}'.format(preds[i], classes[preds[i]]))

        
        try:
            d = QFileDialog()
            d.setFileMode(QFileDialog.AnyFile)
            d.setFilter(QDir.Files)
            if d.exec_():
                fileName = d.selectedFiles()
                #self.TEXT.setText(fileName[0])
                x = predictions(video_dir = fileName[0], model = model, nb_frames = 25, img_size = 224)
                #print(fileName)
                print(x)
        except Exception as e:
            print(str(e))
        
        
        
        
            
    
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "Show Screen"))
        self.pushButton2.setText(_translate("MainWindow", "Upload File"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
