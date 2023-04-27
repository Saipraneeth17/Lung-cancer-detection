
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
import imutils
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
from sklearn import metrics
from pathlib import Path



main = tkinter.Tk()
main.title("Lung Cancer Detection") 
main.geometry("800x600")

global X
global Y
global classifier,classifier1
global filename
#global smplImg

def upload(): 
    global filename
    #filename = filedialog.askdirectory(initialdir=".")
    filename = Path("/LungCancer/sample_dataset_images")
    filename =str(filename)
    text.delete('1.0', END)
    
    #text.insert(END,filename+" loaded\n");

def preprocess():
    global X
    global Y
    global smplImg
    text.delete('1.0', END)
    X = np.load('model/X.txt.npy')
    Y = np.load('model/Y.txt.npy')
    #text.insert(END,"Total process images from dataset is : "+str(X.shape[0])+"\n")
    img = X[0]
    img = cv2.resize(img,(400,400))
    smplImg = img
    
    
    
def showSampleImage():
    cv2.imshow("Sample Image",smplImg)
    cv2.waitKey(0)
    
def buildCNN():
    global X
    global Y
    global classifier,classifier1
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            classifier = model_from_json(loaded_model_json)
        classifier.load_weights("model/model_weights.h5")
        classifier._make_predict_function()

        with open('model/model1.json', "r") as json_file:
            loaded_model_json = json_file.read()
            classifier1 = model_from_json(loaded_model_json)
        classifier1.load_weights("model/model_weights1.h5")
        classifier1._make_predict_function()
        
        print(classifier.summary())
        f = open('model/history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy']
        accuracy = acc[9] * 100
        #text.insert(END,"CNN Lung Cancer Training Accuracy = "+str(accuracy))
    else:
        classifier = Sequential()
        classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Flatten())
        classifier.add(Dense(output_dim = 256, activation = 'relu'))
        classifier.add(Dense(output_dim = 2, activation = 'softmax'))
        print(classifier.summary())
        classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        hist = classifier.fit(X, Y, batch_size=16, epochs=10, shuffle=True, verbose=2)
        classifier.save_weights('model/model_weights.h5')            
        model_json = classifier.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
        f = open('model/history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy']
        accuracy = acc[9] * 100
        #text.insert(END,"CNN Lung Cancer Training Accuracy = "+str(accuracy))

def tumorDetection():
    img = cv2.imread('myimg.png')
    orig = cv2.imread('test1.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    min_area = 0.95*180*35
    max_area = 1.05*180*35
    result = orig.copy()
    for c in contours:
        area = cv2.contourArea(c)
        cv2.drawContours(result, [c], -1, (0, 0, 255), 3)
        if area > min_area and area < max_area:
            cv2.drawContours(result, [c], -1, (0, 255, 255), 3)            
    return result, len(contours)        

def predict():
    global classifier,classifier1
    labels = ['Abnormal','Normal']
    filename = filedialog.askopenfilename(initialdir="testImages")
    image = cv2.imread(filename)
    img = cv2.resize(image, (64,64))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,64,64,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255
    preds = classifier.predict(img)
    predict = np.argmax(preds)
    print(predict)
    img1 = cv2.imread(filename)
    img1 = cv2.resize(img1, (500,400))
    if predict == 0:
        img = cv2.imread(filename,0)
        img = cv2.resize(img,(64,64), interpolation = cv2.INTER_CUBIC)
        img = img.reshape(1,64,64,1)
        img = (img-127.0)/127.0
        preds = classifier1.predict(img)
        preds = preds[0]
        preds = cv2.resize(preds,(300,300),interpolation = cv2.INTER_CUBIC)
        cv2.imwrite("myimg.png",preds*255)
        orig = cv2.imread(filename,0)
        orig = cv2.resize(orig,(300,300),interpolation = cv2.INTER_CUBIC)
        cv2.imwrite("test1.png",orig)
        preds, count = tumorDetection()
        stages = ""
        if count == 1:
            stages = "Stage 1 Cancer Detected"
        elif count == 2:
            stages = "Stage 2 Cancer Detected"
        elif count > 2:
            stages = "Stage 3 Cancer Detected"
        cv2.putText(img1, 'MRI-Scan Predicted as : '+labels[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 255, 255), 2)
        cv2.putText(img1, stages, (10, 55),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 255, 255), 2)
        cv2.imshow('Original Image', img1)
        cv2.imshow("Tumor Detected Image",preds)
        cv2.imshow("Extracted Tumor Region",cv2.imread("myimg.png"))
    else:
        cv2.putText(img1, 'MRI-Scan Predicted as : '+labels[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 255, 255), 2)
        cv2.imshow('MRI-Scan Predicted as : '+labels[predict], img1)
    cv2.waitKey(0)
    
def graph():
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()

    accuracy = data['accuracy']
    loss = data['loss']
    
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy/Loss')
    plt.plot(loss, 'ro-', color = 'red')
    plt.plot(accuracy, 'ro-', color = 'green')
    plt.legend(['Loss', 'Accuracy'], loc='upper left')
    plt.title('CNN Lung Cancer Accuracy & Loss Graph')
    plt.show()


font = ('times', 16, 'bold')
title = Label(main, text='Lung Cancer Detection')
title.config(bg='black', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=-10,y=2)

font1 = ('times', 12, 'bold')
text=Text(main,height=1,width=60)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=300)
text.config(font=font1)
text.config(bg='grey',fg='black',bd=0)


font1 = ('times', 12, 'bold') 

upload() 
#uploadButton = Button(main, text="Upload Lung Cancer Dataset", command=upload)
#uploadButton.place(x=50,y=650)
#uploadButton.config(font=font1)

#preprocessButton = Button(main, text="Preprocess Dataset", command=preprocess)
#preprocessButton.place(x=400,y=550)
#preprocessButton.config(font=font1)
#preprocessButton.config(bg='darkblue',fg='white',bd=0) 
preprocess()

#trainButton = Button(main, text="Get Training Accuracy", command=buildCNN)
#trainButton.place(x=150,y=450)
#trainButton.config(font=font1) 
#trainButton.config(bg='darkblue',fg='white',bd=0)
buildCNN()

showButton = Button(main, text="Show a sample trained image", command=showSampleImage)
showButton.place(x=50,y=200)
showButton.config(font=font1)
showButton.config(bg='darkblue',fg='white',bd=0)

graphButton = Button(main, text="get Accuracy & Loss Graph", command=graph)
graphButton.place(x=560,y=200)
graphButton.config(font=font1)
graphButton.config(bg='darkblue',fg='white',bd=0)

predictButton = Button(main, text="Upload Test Image & Predict Cancer", command=predict)
predictButton.place(x=270,y=200)
predictButton.config(font=font1)
predictButton.config(bg='darkblue',fg='white',bd=0)


main.config(bg='grey')
main.mainloop()
