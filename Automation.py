import sys
import os

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.uic import loadUi

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.fftpack
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import pickle

class PlotCanvas(FigureCanvas):
 
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
 
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
 
        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
 
    def plot(self, value):
        print('called')
        ax = self.figure.add_subplot(111)
        ax.clear()
        ax.plot(value, 'r-')
        ax.set_title('Predicted Result')
        self.draw()

    def matrixPlot(self, value):
        ax = self.figure.add_subplot(111)
        ax.clear()
        ax.matshow(value)
        ax.set_title('Confusion Matrix')
        self.draw()

class Widget(QWidget):
    
    def __init__(self):
        
        print('Training patterns...')
        
        super().__init__()
        uifile = os.path.join(os.path.dirname(__file__), 'Automation.ui')
        self.ui = loadUi(uifile, self)
        
        self.m = PlotCanvas(self, width=4, height=3)
        self.m.move(450,0)

        self.m2 = PlotCanvas(self, width=4, height=3)
        self.m2.move(450,300)
        
        for elem in self.ui.children():
            name = elem.objectName()
            if name == 'btn_classify':
                elem.clicked.connect(self.classify)
            if name == 'btn_internal':
                elem.clicked.connect(self.internalPredict)
            if name == 'btn_test':
                elem.clicked.connect(self.loadTestFile)
            if name == 'btn_finalTest':
                elem.clicked.connect(self.plotFinalGraph)
            if name == 'btn_load':
                elem.clicked.connect(self.loadModal)
            if name == 'txt_accuracy':
                self.txtAccuracy = elem
            if name == 'txt_test':
                self.txtTest = elem
              

    def loadModal(self): 
        self.mainInternalTest= np.loadtxt('testData.txt')
        self.mainInteralTarget= np.loadtxt('testTarget.txt')

        with open('classifierModal.pkl', 'rb') as f:
          self.clf = pickle.load(f)
          print(self.clf.predict)
          

    def splitData(self, data):
      if(len(data)%3 == 0):
        data1, data2, test1 = np.vsplit(data, 3)
        concatData = np.concatenate((data1, data2), axis=0)
        return concatData, test1

      if(len(data)%3 == 1):
        cpyData = data[:-1]
        data1, data2, test1 = np.vsplit(cpyData, 3)
        concatData = np.concatenate((data1, data2), axis=0)
        return concatData, test1
      
      cpyData = data[:-2]
      data1, data2, test1 = np.vsplit(cpyData, 3)
      concatData = np.concatenate((data1, data2), axis=0)
      return concatData, test1

    def loadFile11(self):
        self.data1part1 = pd.read_excel("State 1 File 1 Dezimalkomma.xlsx", header=None)
        self.data1part1 = self.data1part1.as_matrix()
        print('loaded')

        # np.random.shuffle(self.data1part1)
        # self.data1 = self.data1part1
        # self.data1, self.test1 = self.splitData(self.data1part1)
        print('loaded')
    
    def loadFile12(self):
        self.data1part2 = pd.read_excel("State 1 File 2 Dezimalkomma.xlsx", header=None)
        self.data1part2 = self.data1part2.as_matrix()
        concatData = np.concatenate((self.data1part1, self.data1part2), axis=0)
        np.random.shuffle(concatData)
        self.data1, self.test1 = self.splitData(concatData)
        print('loaded')

    def loadFile2(self):
        data = pd.read_excel("State 2 Dezimalkomma_calc.xlsx", header=None)
        data = data.as_matrix()
        np.random.shuffle(data)
        self.data2, self.test2 = self.splitData(data)
        print('loaded')

    def classify(self):
      try:
        self.loadFile11();
        self.loadFile12();
        self.loadFile2();

        self.internal_extract_features()
        self.clf = DecisionTreeClassifier(criterion='entropy')
        self.clf = self.clf.fit(self.classifierData, self.classifierTarget)

        with open('classifierModal.pkl', 'wb') as f:
          pickle.dump(self.clf, f)
        mat = np.matrix(self.mainInteralTarget)

        with open('testTarget.txt','wb') as f:
            for line in mat:
                np.savetxt(f, line, fmt='%.2f')
        
        mat = np.matrix(self.mainInternalTest)
        with open('testData.txt','wb') as f:
            for line in mat:
                np.savetxt(f, line, fmt='%.2f')
      except Exception:
        print(Exception) 
        pass

    def internalPredict(self): 
      result = self.clf.predict(self.mainInternalTest)
      print(result)
      matrix = confusion_matrix(self.mainInteralTarget, result) 
      print(matrix)     
      self.m.matrixPlot(matrix)
      
      self.txt_accuracy.setText(str(accuracy_score(self.mainInteralTarget, result)))

    def generateDataAndTarget(self, data, targetValue):
        fft = scipy.fftpack.fft(data)
        abs = np.absolute(fft)

        # features 1
        rms = np.sqrt(np.mean(abs, axis=1))
        median = np.median(abs, axis=1)
        var = np.var(abs, axis=1)
        min_x = np.amin(abs, axis=1)

        data = [[]]
        target = [[]]
        for x in range(0, len(rms)):
            data.append([])
            target.append([])
            data[x].append(var[x])
            data[x].append(rms[x])
            data[x].append(min_x[x])
            data[x].append(median[x])
            target[x].append(targetValue)
        data.remove([])
        target.remove([])
        return data, target
    
    def internal_extract_features(self):
      try:
        shuffleData1, shuffleTarget1 = self.generateDataAndTarget(self.data1, 0)
        shuffleData2, shuffleTarget2 = self.generateDataAndTarget(self.data2, 1)
        self.classifierData = np.concatenate((shuffleData1, shuffleData2), axis=0)
        self.classifierTarget = np.concatenate((shuffleTarget1, shuffleTarget2), axis=0)

        testData1, testTarget1 = self.generateDataAndTarget(self.test1, 0)
        testData2, testTarget2 = self.generateDataAndTarget(self.test2, 1)

        self.mainInternalTest = np.concatenate((testData1, testData2), axis=0)
        self.mainInteralTarget = np.concatenate((testTarget1, testTarget2), axis=0)
      except Exception:
        print(Exception) 
        pass

    def loadTestFile(self):
      try:
        filename = QFileDialog.getOpenFileName(w, 'Open File', './')
        data = pd.read_excel(filename[0], header=None)
        self.txtTest.setText(filename[0])
        self.testFinalData = data.as_matrix()
      except Exception: 
        pass

    def plotFinalGraph(self):
      try:
        shuffleData1, shuffleTarget1 = self.generateDataAndTarget(self.testFinalData, 0)
        print(self.clf.predict(shuffleData1))
        self.m2.plot(self.clf.predict(shuffleData1))
      except Exception: 
        print(Exception)
        pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Widget()
    w.show()
    sys.exit(app.exec_())
