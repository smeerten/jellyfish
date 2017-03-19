import numpy as np
import scipy.signal
import os

import sip
import sys
sip.setapi('QString', 2)
try:
    import bla
    from PyQt4 import QtGui, QtCore
    from PyQt4 import QtGui as QtWidgets
    QT = 4
except ImportError:
    from PyQt5 import QtGui, QtCore, QtWidgets
    QT = 5
import matplotlib
if QT ==4:
    matplotlib.use('Qt4Agg')
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
else:
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from spectrumFrame import Plot1DFrame
from safeEval import safeEval










GAMMASCALE = 42.577469 / 100
with open(os.path.dirname(os.path.realpath(__file__)) +"/IsotopeProperties") as isoFile:
    isoList = [line.strip().split('\t') for line in isoFile]
isoList = isoList[1:]
N = len(isoList)
#nameList = []
#fullNameList = []
ABBREVLIST = []
atomNumList = np.zeros(N)
atomMassList = np.zeros(N)
spinList = np.zeros(N)
abundanceList = np.zeros(N)
gammaList = np.zeros(N)
freqRatioList = np.zeros(N)

for i in range(N):
    isoN = isoList[i]
    if isoN[3] != '-' and isoN[4] != '-' and isoN[8] != '-':
        atomNumList[i] = int(isoN[0])
#        nameList = np.append(nameList, isoN[1])
#        fullNameList = np.append(fullNameList, isoN[2])
        atomMassList[i] = int(isoN[3])
        ABBREVLIST.append( str(int(atomMassList[i])) + isoN[1])
        spinList[i] = isoN[4]
        if isoN[5] == '-':
            abundanceList[i] = 0
        else:
            abundanceList[i] = isoN[5]
        freqRatioList[i] = isoN[8]







class spinCls:
    def __init__(self, Nucleus, shift, Detect):
        self.index = ABBREVLIST.index(Nucleus)
        self.I = spinList[self.index]
        self.Gamma = freqRatioList[self.index] * GAMMASCALE * 1e6
        self.shift = shift
        self.Detect = Detect
        self.m = np.linspace(self.I,-self.I,self.I*2+1)
        self.Iarray = np.linspace(self.I,self.I,self.I*2+1)
        self.Iz = np.diag(self.m)
        self.Iplus = np.diag(np.sqrt(self.I*(self.I+1)-self.m*(self.m+1))[1:],1)
        self.Imin = np.diag(np.diag(self.Iplus,1),-1)
        self.Ix = 0.5 * (self.Iplus + self.Imin)
        self.Iy = -0.5j * (self.Iplus - self.Imin)
        
        self.Ident = np.eye(int(self.I*2+1))

        
class spinSystemCls:
    def __init__(self, SpinList, Jmatrix, B0,HighOrder = True):      
        self.SpinList = SpinList
        self.Jmatrix = Jmatrix
        self.B0 = B0
        self.HighOrder = HighOrder
        self.GetMatrixSize()
        self.OperatorsFunctions = {'Iz': lambda Spin: Spin.Iz , 'Ix': lambda Spin: Spin.Ix, 'Iy': lambda Spin: Spin.Iy}
        self.SpinOperators = {}
        for Operator in self.OperatorsFunctions.keys():
            self.SpinOperators[Operator] = self.MakeSingleOperator(self.OperatorsFunctions[Operator])

        self.Htot = self.MakeJhamiltonian() + self.MakeShiftHamil()
        self.DetectOp = self.MakeDetect()
        self.RhoZero = self.MakeRhoZero()
        
    def GetMatrixSize(self):
        self.MatrixSize = 1
        for spin in self.SpinList:
            self.MatrixSize = int(self.MatrixSize * (spin.I * 2 + 1))
            
            
    def MakeSingleOperator(self,Operator):
        Matrix = np.zeros((len(self.SpinList),self.MatrixSize,self.MatrixSize),dtype=complex)
        for spin in  range(len(self.SpinList)):
            IList = []
            for subspin in range(len(self.SpinList)):
                if spin == subspin:
                    IList.append(Operator(self.SpinList[subspin]))
                else:
                    IList.append(self.SpinList[subspin].Ident)
            
            Matrix[spin,:,:] = self.kronList(IList)
        return Matrix
    

    def MakeMultipleOperator(self,Operator,SelectList):
       IList = []
       for spin in  range(len(self.SpinList)):
           if spin in SelectList:
               IList.append(Operator(self.SpinList[spin]))
           else:
               IList.append(self.SpinList[spin].Ident)
       Matrix = self.kronList(IList)
       return Matrix
       
   
    def kronList(self,List):
        M = 1
        for element in List:
            M = np.kron(M , element)
        return M
        
    def MakeShiftHamil(self):
        Shift = np.zeros((self.MatrixSize,self.MatrixSize))
        for index in range(len(self.SpinList)):
            Shift = Shift + (self.SpinList[index].shift * 1e-6 + 1) * self.SpinList[index].Gamma * self.B0 * self.SpinOperators['Iz'][index]
        return Shift
        
    def MakeJhamiltonian(self):
        Jmatrix = self.Jmatrix
        if self.HighOrder:
            OperatorsFunctions = {'Iz': lambda Spin: Spin.Iz , 'Ix': lambda Spin: Spin.Ix, 'Iy': lambda Spin: Spin.Iy}
        else:
            OperatorsFunctions = {'Iz': lambda Spin: Spin.Iz}
        Jham = np.zeros((self.MatrixSize,self.MatrixSize))
        for spin in range(len(self.SpinList)):
            for subspin in range(spin,len(self.SpinList)):
                    if Jmatrix[spin,subspin] != 0:
                        temp = np.zeros((self.MatrixSize,self.MatrixSize),dtype=complex)
                        for operator in OperatorsFunctions.keys():
                            temp += self.MakeMultipleOperator(OperatorsFunctions[operator],[spin,subspin])
                        Jham = Jham + Jmatrix[spin,subspin] * temp
        return Jham
                        
        
    def MakeDetect(self):
        Detect = np.zeros((self.MatrixSize,self.MatrixSize))
        for index in range(len(self.SpinList)):
            if self.SpinList[index].Detect:
               Detect =  Detect + self.SpinOperators['Ix'][index] + 1J * self.SpinOperators['Iy'][index]
        return Detect


    def MakeRhoZero(self):
        RhoZero = np.zeros((self.MatrixSize,self.MatrixSize))
        for index in range(len(self.SpinList)):
            if self.SpinList[index].Detect:
               RhoZero =  RhoZero + self.SpinOperators['Ix'][index]
        return RhoZero



def MakeSpectrum(SpinSystem,RefFreq,B0,AxisLimits,LineBroadening,NumPoints):
    Limits = tuple(AxisLimits * RefFreq * 1e-6)
    sw = Limits[1] - Limits[0]
    dw = 1.0/ sw
    lb = LineBroadening
    
    #Make propagators
    Hdiag,T = np.linalg.eigh(SpinSystem.Htot)
    Tinv = np.linalg.inv(T)
    
    RhoProp = Tinv @ SpinSystem.RhoZero @ T
    RhoProp =  np.tril(RhoProp,1)

    DetectProp = np.real(Tinv @ SpinSystem.DetectOp @ T)
    AllInts = np.real(DetectProp * RhoProp)
    
    #Get intensies and frequencies
    Intensities = []
    Frequencies = []

    for iii in range(RhoProp.shape[0]):
        for jjj in range(iii):
            if abs(RhoProp[iii,jjj]) > 1e-9:
                Intensities.append(AllInts[iii,jjj])
                Frequencies.append(Hdiag[iii] - Hdiag[jjj] - RefFreq)

    #Make spectrum
    Spectrum, Axis = np.histogram(Frequencies, int(NumPoints), Limits , weights = Intensities)
    
    if np.sum(np.isnan(Spectrum)):
       Spectrum = np.zeros_like(Axis)
    elif np.max(Spectrum) == 0.0:
        pass
    else:
       Fid = np.fft.ifft(np.fft.ifftshift(np.conj(scipy.signal.hilbert(Spectrum))))
       TimeAxis = np.linspace(0,NumPoints-1,NumPoints) * dw
       Fid = Fid * np.exp(-TimeAxis * lb)
   
       Spectrum = np.real(np.fft.fftshift(np.fft.fft(Fid)))
#       Spectrum = Spectrum / np.max(Spectrum)

    Axis = (Axis[1:] + 0.5 * (Axis[0] - Axis[1]))  / (RefFreq * 1e-6)
    
    return Spectrum, Axis, RefFreq
    





class PlotFrame(Plot1DFrame):

    def __init__(self, root, fig, canvas):
        super(PlotFrame, self).__init__(root, fig, canvas)
        self.canvas.mpl_connect('button_press_event', self.buttonPress)
        self.canvas.mpl_connect('button_release_event', self.buttonRelease)
        self.canvas.mpl_connect('motion_notify_event', self.pan)
        self.canvas.mpl_connect('scroll_event', self.scroll)
        self.canvas.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.canvas.setFocus()
        self.xmaxlim= None
        self.xminlim= None
        self.ymaxlim= None
        self.yminlim= None

    def setData(self, xdata, ydata):
        self.xdata = xdata
        self.ydata = ydata

    def plotReset(self, xReset=True, yReset=True):  # set the plot limits to min and max values
        miny = min(self.ydata)
        maxy = max(self.ydata)
        differ = 0.05 * (maxy - miny)  # amount to add to show all datapoints (10%)
        if yReset:
            self.yminlim = miny - differ
            self.ymaxlim = maxy + differ
        axMult = 1.0 
        if xReset:
            self.xminlim = min(self.xdata * axMult)
            self.xmaxlim = max(self.xdata * axMult)
        self.ax.set_xlim(self.xmaxlim, self.xminlim)
        self.ax.set_ylim(self.yminlim, self.ymaxlim)
        
    def showFid(self):
        self.ax.cla()
        self.ax.plot(self.xdata, self.ydata)
        if self.xmaxlim is None:
            self.plotReset()
        self.ax.set_xlim(self.xmaxlim, self.xminlim)
        self.ax.set_ylim(self.yminlim, self.ymaxlim)
        self.canvas.draw()        


class SettingsFrame(QtWidgets.QWidget):

    def __init__(self, parent):
        super(SettingsFrame, self).__init__(parent)
        self.father = parent
        grid = QtWidgets.QGridLayout(self)
        grid.addWidget(QtWidgets.QLabel("B0 [T]:"), 0, 0,QtCore.Qt.AlignHCenter)
        self.B0Setting = QtWidgets.QLineEdit(self)
        self.B0Setting.setAlignment(QtCore.Qt.AlignHCenter)
        self.B0Setting.setText(str(self.father.B0))
        self.B0Setting.returnPressed.connect(self.ApplySettings)
        grid.addWidget(self.B0Setting, 0, 1)
        
        self.LbType = QtWidgets.QComboBox()
        self.LbType.addItems(['Line Width [Hz]:','Line Width [ppm]:'])
        self.LbType.currentIndexChanged.connect(self.ChangeLbSetting)
        grid.addWidget(self.LbType, 1, 0)
        self.LbSetting = QtWidgets.QLineEdit(self)
        self.LbSetting.setAlignment(QtCore.Qt.AlignHCenter)
        self.LbSetting.setText(str(self.father.Lb))
        self.LbSetting.returnPressed.connect(self.ApplySettings)
        grid.addWidget(self.LbSetting, 1, 1)
        
        grid.addWidget(QtWidgets.QLabel("# Points [x1024]:"), 0, 2,QtCore.Qt.AlignHCenter)
        self.NumPointsSetting = QtWidgets.QLineEdit(self)
        self.NumPointsSetting.setAlignment(QtCore.Qt.AlignHCenter)
        self.NumPointsSetting.setText(str(self.father.NumPoints/1024))
        self.NumPointsSetting.returnPressed.connect(self.ApplySettings)
        grid.addWidget(self.NumPointsSetting, 0, 3)
           
        grid.addWidget(QtWidgets.QLabel("Ref Nucleus:"), 1, 2,QtCore.Qt.AlignHCenter)
        self.RefNucleusSettings = QtWidgets.QComboBox()
        self.RefNucleusSettings.addItems(ABBREVLIST)
        self.RefNucleusSettings.setCurrentIndex(0)
        self.RefNucleusSettings.currentIndexChanged.connect(self.ApplySettings)
        grid.addWidget(self.RefNucleusSettings, 1, 3)
        
        
        grid.addWidget(QtWidgets.QLabel("x Min [ppm]:"), 0, 4,QtCore.Qt.AlignHCenter)
        self.XminSetting = QtWidgets.QLineEdit(self)
        self.XminSetting.setAlignment(QtCore.Qt.AlignHCenter)
        self.XminSetting.setText(str(self.father.Limits[0]))
        self.XminSetting.returnPressed.connect(lambda: self.ApplySettings(True))
        grid.addWidget(self.XminSetting, 0, 5)
        
        grid.addWidget(QtWidgets.QLabel("x Max [ppm]:"), 1, 4,QtCore.Qt.AlignHCenter)
        self.XmaxSetting = QtWidgets.QLineEdit(self)
        self.XmaxSetting.setAlignment(QtCore.Qt.AlignHCenter)
        self.XmaxSetting.setText(str(self.father.Limits[1]))
        self.XmaxSetting.returnPressed.connect(lambda: self.ApplySettings(True))
        grid.addWidget(self.XmaxSetting, 1, 5)
        
        grid.setColumnStretch(10, 1)
        grid.setRowStretch(10, 1)
    
    def ChangeLbSetting(self):
        if self.LbType.currentIndex() == 0: #From ppm
            self.LbSetting.setText(str(safeEval(self.LbSetting.text()) * (self.father.RefFreq * 1e-6)))
        else:
            self.LbSetting.setText(str(safeEval(self.LbSetting.text()) / (self.father.RefFreq * 1e-6)))
        
        
        
    def ApplySettings(self,ResetAxis = False):
        self.father.B0 = safeEval(self.B0Setting.text())
        self.father.RefNucleus = ABBREVLIST[self.RefNucleusSettings.currentIndex()]
        self.father.SetRefFreq()
        if self.LbType.currentIndex() == 0:
            self.father.Lb = safeEval(self.LbSetting.text())
        else:
            self.father.Lb = safeEval(self.LbSetting.text()) * (self.father.RefFreq * 1e-6)
        self.father.NumPoints = safeEval(self.NumPointsSetting.text()) * 1024
        
        self.father.Limits[0] = float(self.XminSetting.text())
        self.father.Limits[1] = float(self.XmaxSetting.text())
        self.father.sim(ResetAxis)



class SpinsysFrame(QtWidgets.QWidget):

    def __init__(self, parent):
        super(SpinsysFrame, self).__init__(parent)
        self.father = parent
        self.grid = QtWidgets.QGridLayout(self)
        self.grid.addWidget(QtWidgets.QLabel("Spin System:"), 0, 0,1,5,QtCore.Qt.AlignHCenter)

        self.addButton = QtWidgets.QPushButton("Add isotope")
        self.addButton.clicked.connect(self.addIsotopeManager)

        self.grid.addWidget(self.addButton,1,0,1,5)
        
        self.setJButton = QtWidgets.QPushButton("Set J-couplings")
        self.setJButton.clicked.connect(self.setJManager)
        self.grid.addWidget(self.setJButton,2,0,1,5)

        self.grid.addWidget(QtWidgets.QLabel("#:"), 5, 0,QtCore.Qt.AlignHCenter)
        self.grid.addWidget(QtWidgets.QLabel("Type:"), 5, 1,QtCore.Qt.AlignHCenter)
        self.grid.addWidget(QtWidgets.QLabel("Shift:"), 5, 2,QtCore.Qt.AlignHCenter)
        self.grid.addWidget(QtWidgets.QLabel("Multiplicity:"), 5, 3,QtCore.Qt.AlignHCenter)
        self.grid.addWidget(QtWidgets.QLabel("Remove:"), 5, 4,QtCore.Qt.AlignHCenter)
        self.spinSysWidgets = {'Number':[],'Isotope':[], 'Shift':[], 'Multi':[]}
        self.Nspins = 0
        self.grid.setColumnStretch(10, 1)
        self.grid.setRowStretch(10, 1)
        
        self.addSpin('1H',1.23,1)
        self.addSpin('13C',10.2,4)
        
    def addSpin(self,Isotope,Shift,Multiplicity):
        self.Nspins += 1
        self.spinSysWidgets['Number'].append(QtWidgets.QLabel(str(self.Nspins)))
        self.spinSysWidgets['Isotope'].append(QtWidgets.QLabel(Isotope))
        
        self.grid.addWidget(self.spinSysWidgets['Isotope'][-1],5 + self.Nspins,1)
        self.grid.addWidget(self.spinSysWidgets['Number'][-1],5 + self.Nspins,0)
        
        
        self.spinSysWidgets['Shift'].append(QtWidgets.QLineEdit())
        self.spinSysWidgets['Shift'][-1].setText(str(Shift))
        self.spinSysWidgets['Shift'][-1].returnPressed.connect(self.parseSpinSys)
        self.grid.addWidget(self.spinSysWidgets['Shift'][-1],5 + self.Nspins,2)
        self.spinSysWidgets['Multi'].append(QtWidgets.QSpinBox())
        self.spinSysWidgets['Multi'][-1].setValue(Multiplicity)
        self.spinSysWidgets['Multi'][-1].setMinimum(1)
        self.grid.addWidget(self.spinSysWidgets['Multi'][-1],5 + self.Nspins,3)
        
        self.parseSpinSys(True)
     
        
    def setJManager(self):
        dialog = setJWindow(self)
        if dialog.exec_():
            if dialog.closed:
                return
            else:
                pass
#                self.addSpin(dialog.Isotope,dialog.Shift,dialog.Multi)
                
    def addIsotopeManager(self):
        dialog = addIsotopeWindow(self)
        if dialog.exec_():
            if dialog.closed:
                return
            else:
                self.addSpin(dialog.Isotope,dialog.Shift,dialog.Multi)

                
    def parseSpinSys(self,ResetAxis = False):
        self.father.SpinList = []
        NSpins = len(self.spinSysWidgets['Isotope'])
        
        for Spin in range(NSpins):
            self.father.SpinList.append(spinCls(self.spinSysWidgets['Isotope'][Spin].text(),safeEval(self.spinSysWidgets['Shift'][Spin].text()),True))
        
        self.father.Jmatrix = np.zeros((NSpins,NSpins))
        
        self.father.sim(ResetAxis,ResetAxis)
            
    def drawSpinSys(self):
        for Spin in range(self.spinSystem['Isotope']):
            self.spinSysWidgets['Isotope']
        
class addIsotopeWindow(QtWidgets.QDialog):
    def __init__(self, parent):
        QtWidgets.QWidget.__init__(self, parent)
        self.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.Tool)
        self.father = parent
        self.Isotope = ''
        self.Shift = 0
        self.Multi = 0
        self.closed = False
        self.setWindowTitle("Add Isotope")
        grid = QtWidgets.QGridLayout(self)
        grid.addWidget(QtWidgets.QLabel("Type:"), 0, 0,QtCore.Qt.AlignHCenter)
        grid.addWidget(QtWidgets.QLabel("Shift [ppm]:"), 0, 1,QtCore.Qt.AlignHCenter)
        grid.addWidget(QtWidgets.QLabel("Multiplicity:"), 0, 2,QtCore.Qt.AlignHCenter)
        
        self.typeSetting = QtWidgets.QComboBox()
        self.typeSetting.addItems(ABBREVLIST)
        self.typeSetting.setCurrentIndex(0)
        grid.addWidget(self.typeSetting,1,0)
        
        self.shiftSetting = QtWidgets.QLineEdit()
        self.shiftSetting.setText(str(0))
        grid.addWidget(self.shiftSetting,1,1)
        
        self.multiSettings = QtWidgets.QSpinBox()
        self.multiSettings.setValue(1)
        self.multiSettings.setMinimum(1)
        grid.addWidget(self.multiSettings,1,2)
        
        cancelButton = QtWidgets.QPushButton("&Cancel")
        cancelButton.clicked.connect(self.closeEvent)
        grid.addWidget(cancelButton, 13, 0)
        okButton = QtWidgets.QPushButton("&Ok")
        okButton.clicked.connect(self.applyAndClose)
        grid.addWidget(okButton, 13, 2)
        
        self.show()
        self.setFixedSize(self.size())
        
    def closeEvent(self, *args):
        self.closed = True
        self.accept()
        self.deleteLater()

    def applyAndClose(self):
        self.Isotope = ABBREVLIST[self.typeSetting.currentIndex()]
        self.Shift = safeEval(self.shiftSetting.text())
        self.Multi = self.multiSettings.value()
        
        self.accept()
        self.deleteLater()

class setJWindow(QtWidgets.QDialog):
    def __init__(self, parent, Jmatrix, numSpins):
        QtWidgets.QWidget.__init__(self, parent)
        self.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.Tool)
        self.father = parent
        self.closed = False
        self.Jmatrix = Jmatrix
        self.numSpins = numSpins
        grid = QtWidgets.QGridLayout(self)
        cancelButton = QtWidgets.QPushButton("&Cancel")
        cancelButton.clicked.connect(self.closeEvent)
        grid.addWidget(cancelButton, 13, 0)
        okButton = QtWidgets.QPushButton("&Ok")
        okButton.clicked.connect(self.applyAndClose)
        grid.addWidget(okButton, 13, 2)
        
        self.show()
        self.setFixedSize(self.size())
        
    def closeEvent(self, *args):
        self.closed = True
        self.accept()
        self.deleteLater()

    def applyAndClose(self):
#        self.Isotope = ABBREVLIST[self.typeSetting.currentIndex()]
#        self.Shift = safeEval(self.shiftSetting.text())
#        self.Multi = self.multiSettings.value()
        
        self.accept()
        self.deleteLater()
        
class MainProgram(QtWidgets.QMainWindow):

    def __init__(self, root):
        super(MainProgram, self).__init__()
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.main_widget = QtWidgets.QWidget(self)
        self.mainFrame = QtWidgets.QGridLayout(self.main_widget)
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.gca()
        self.mainFrame.addWidget(self.canvas, 0, 0)
        self.mainFrame.setColumnStretch(0, 1)
        self.mainFrame.setRowStretch(0, 1)
        
        self.B0 = 14.1 #Tesla
        self.Lb = 1 #Hz
        self.NumPoints = 1024*32
        self.Limits = np.array([0.0,8.0]) #ppm
        self.RefNucleus = '1H'
        self.RefFreq = 0
        self.SetRefFreq()
        self.Jmatrix = None
        self.StrongCoupling = True
        self.SpinList = []
        self.Jmatrix = np.zeros((len(self.SpinList),len(self.SpinList)))
        self.PlotFrame = PlotFrame(self, self.fig, self.canvas)
        self.settingsFrame = SettingsFrame(self)
        self.mainFrame.addWidget(self.settingsFrame, 1, 0)
        self.spinsysFrame = SpinsysFrame(self)
        self.mainFrame.addWidget(self.spinsysFrame, 0, 1,2,1)
     

        
        
        self.sim()
        
    def SetRefFreq(self):
        index = ABBREVLIST.index(self.RefNucleus)
        self.RefFreq = freqRatioList[index] * GAMMASCALE * 1e6 * self.B0
        
    def sim(self,ResetXAxis = False, ResetYAxis = False):
        if self.Jmatrix is None:
            self.Jmatrix = np.zeros((len(self.SpinList),len(self.SpinList)))
        self.SpinSystem = spinSystemCls(self.SpinList, self.Jmatrix, self.B0,self.StrongCoupling)
        self.Spectrum, self.Axis, self.RefFreq = MakeSpectrum(self.SpinSystem,self.RefFreq,self.B0,self.Limits,self.Lb,self.NumPoints)
        self.PlotFrame.setData(self.Axis, self.Spectrum)
        if ResetXAxis:
            self.PlotFrame.plotReset(xReset = True, yReset = False)
        if ResetYAxis:
            self.PlotFrame.plotReset(xReset = False, yReset = True)
        self.PlotFrame.showFid()
   

if __name__ == '__main__':
    root = QtWidgets.QApplication(sys.argv)
    mainProgram = MainProgram(root)
    mainProgram.setWindowTitle("Jellyfish")
    mainProgram.show()
    sys.exit(root.exec_())

    

        
   











