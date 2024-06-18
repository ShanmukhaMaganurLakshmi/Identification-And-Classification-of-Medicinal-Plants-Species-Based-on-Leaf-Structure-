import tkinter as tk
from tkinter import Message ,Text
from PIL import Image, ImageTk
import pandas as pd
import tkinter.ttk as ttk
import tkinter.font as font
import tkinter.messagebox as tm
import matplotlib.pyplot as plt
import numpy as np
from tkinter import filedialog
import tkinter.messagebox as tm
import argparse
import cv2
import sys
import glob
import math
import time
import os
import training as TR
import test as PR

bgcolor="#DAF7A6"
bgcolor1="#B7C526"
fgcolor="black"
def Home():
        global window
        def clear():
            print("Clear1")
            txt.delete(0, 'end')    
            txt1.delete(0, 'end')    
            


        window = tk.Tk()
        window.title("Plant Leaf Identification Using Deep Learning")

 
        window.geometry('1280x720')
        window.configure(background=bgcolor)
        #window.attributes('-fullscreen', True)

        window.grid_rowconfigure(0, weight=1)
        window.grid_columnconfigure(0, weight=1)
        

        message1 = tk.Label(window, text="Plant Leaf  Identification Using Deep Learning" ,bg=bgcolor  ,fg=fgcolor  ,width=50  ,height=3,font=('times', 30, 'italic bold underline')) 
        message1.place(x=100, y=20)

        lbl = tk.Label(window, text="Select Dataset",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl.place(x=100, y=200)
        
        txt = tk.Entry(window,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt.place(x=400, y=215)


        lbl1 = tk.Label(window, text="Select Image",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl1.place(x=100, y=270)
        
        txt1 = tk.Entry(window,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt1.place(x=400, y=275)
        
        def browse():
                path=filedialog.askdirectory()
                print(path)
                txt.delete(0, 'end')
                txt.insert('end',path)
                if path !="":
                        print(path)
                else:
                        tm.showinfo("Input error", "Select DataSet Folder")     

        def browse1():
                path=filedialog.askopenfilename()
                print(path)
                txt1.delete(0, 'end')
                txt1.insert('end',path)
                if path !="":
                        print(path)
                else:
                        tm.showinfo("Input error", "Select Input Image")        

        def Trainprocess():
                sym=txt.get()
                if sym != "":
                        TR.process(sym)
                        tm.showinfo("Output", "Training Successfully Finished")
                else:
                        tm.showinfo("Input error", "Select Dataset Folder")

        def Predictprocess():
                sym=txt1.get()
                if sym != "" :
                        res,conf=PR.process(sym)
                        
                        tm.showinfo("Output", "Identified as :   "+str(res)+" With Confidance Score of "+str(conf))
                else:
                        tm.showinfo("Input error", "Select Input Image")
        

        browse = tk.Button(window, text="Browse", command=browse  ,fg=fgcolor  ,bg=bgcolor1  ,width=20  ,height=1, activebackground = "Red" ,font=('times', 15, ' bold '))
        browse.place(x=650, y=200)

        browse1 = tk.Button(window, text="Browse", command=browse1  ,fg=fgcolor  ,bg=bgcolor1  ,width=20  ,height=1, activebackground = "Red" ,font=('times', 15, ' bold '))
        browse1.place(x=650, y=270)

        clearButton = tk.Button(window, text="Clear", command=clear  ,fg=fgcolor  ,bg=bgcolor1  ,width=20  ,height=2 ,activebackground = "Red" ,font=('times', 15, ' bold '))
        clearButton.place(x=950, y=200)
         
        

        TRbutton = tk.Button(window, text="Training", command=Trainprocess  ,fg=fgcolor   ,bg=bgcolor1   ,width=20  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
        TRbutton.place(x=400, y=600)


        PRbutton = tk.Button(window, text="Prediction", command=Predictprocess  ,fg=fgcolor   ,bg=bgcolor1 ,width=15  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
        PRbutton.place(x=620, y=600)

        quitWindow = tk.Button(window, text="Quit", command=window.destroy  ,fg=fgcolor   ,bg=bgcolor1  ,width=15  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
        quitWindow.place(x=800, y=600)

        window.mainloop()
Home()

