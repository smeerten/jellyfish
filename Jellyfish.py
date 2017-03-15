import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import time



class spinCls:
    def __init__(self, I, shift, Gamma, Detect):
        self.I = I
        self.shift = shift
        self.Gamma = Gamma
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
        for spin in SpinList:
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
        for index in range(len(SpinList)):
            Shift = Shift + SpinList[index].shift * 1e-6 * SpinList[index].Gamma * self.B0 * self.SpinOperators['Iz'][index]
        return Shift
        
    def MakeJhamiltonian(self):
        Jmatrix = self.Jmatrix
        if self.HighOrder:
            OperatorsFunctions = {'Iz': lambda Spin: Spin.Iz , 'Ix': lambda Spin: Spin.Ix, 'Iy': lambda Spin: Spin.Iy}
        else:
            OperatorsFunctions = {'Iz': lambda Spin: Spin.Iz}
        Jham = np.zeros((self.MatrixSize,self.MatrixSize))
        for spin in range(len(self.SpinList)):
            for subspin in range(len(self.SpinList)):
                if subspin > spin:
                    if Jmatrix[spin,subspin] != 0:
                        temp = np.zeros((self.MatrixSize,self.MatrixSize),dtype=complex)
                        for operator in OperatorsFunctions.keys():
                            temp += self.MakeMultipleOperator(OperatorsFunctions[operator],[spin,subspin])
                        Jham = Jham + Jmatrix[spin,subspin] * temp
        return Jham
                        
        
    def MakeDetect(self):
        Detect = np.zeros((self.MatrixSize,self.MatrixSize))
        for index in range(len(SpinList)):
            if SpinList[index].Detect:
               Detect =  Detect + self.SpinOperators['Ix'][index] + 1J * self.SpinOperators['Iy'][index]
        return Detect


    def MakeRhoZero(self):
        RhoZero = np.zeros((self.MatrixSize,self.MatrixSize))
        for index in range(len(SpinList)):
            if SpinList[index].Detect:
               RhoZero =  RhoZero + self.SpinOperators['Ix'][index]
        return RhoZero

temp = time.time()
B0 = 2.10 #Tesla
SpinA = spinCls(0.5,1.23,42.576e6,True)
SpinB = spinCls(0.5,3.69,42.576e6,True)

SpinList = [SpinA,SpinA,SpinA,SpinB,SpinB,SpinB,SpinB,SpinB,SpinB,SpinB]
Jmatrix = np.zeros((len(SpinList),len(SpinList)))

Jmatrix[0,3] = 7 
Jmatrix[0,4] = 7 
Jmatrix[0,5] = 7 
Jmatrix[0,6] = 7 
Jmatrix[0,7] = 7 
Jmatrix[0,8] = 7 
Jmatrix[0,9] = 7
Jmatrix[1,3] = 7 
Jmatrix[1,4] = 7 
Jmatrix[1,5] = 7 
Jmatrix[1,6] = 7 
Jmatrix[1,7] = 7 
Jmatrix[1,8] = 7 
Jmatrix[1,9] = 7
Jmatrix[2,3] = 7 
Jmatrix[2,4] = 7
Jmatrix[2,5] = 7 
Jmatrix[2,6] = 7 
Jmatrix[2,7] = 7 
Jmatrix[2,8] = 7
Jmatrix[2,9] = 7 

SpinSystem = spinSystemCls(SpinList, Jmatrix, B0,True)
print('Spin System = ' + str(time.time() - temp) + ' s')


    
   


#Settings========
dw = 10e-4
sw = 1.0/dw
points = 1024*32
process = 1024*128
axis = np.linspace(0,points-1,points) * dw
lb = 1 #Hz
freqaxis =  np.fft.fftshift(np.fft.fftfreq(process, d=dw))/(42.576 * B0)

RhoZero = SpinSystem.RhoZero
Htot = SpinSystem.Htot
Detect = SpinSystem.DetectOp
#Diagonalize========
temp = time.time()
Hdiag,T = np.linalg.eigh(Htot)
Tinv = np.linalg.inv(T)
print('Diag = ' + str(time.time() - temp) + ' s')

#PropL = T @ np.diag(np.exp(-1j*2*np.pi*Hdiag*dw)) @  np.linalg.inv(T)
#PropR = np.conj(PropL)
#print(time.time() - a)
#
##Simulate
#res = []
#Rho = RhoZero
#for iii in range(points):
#    res.append(np.trace(Rho @ Detect))
#    Rho = PropL @ Rho @ PropR
#
#
## Process=========  
#res[0] = res[0]/2
#res = np.array(res) * np.exp(-axis * lb)
#Spectrum = np.real(np.fft.fftshift(np.fft.fft(res,process)))
#Spectrum = Spectrum /np.max(Spectrum)
#print(time.time() - a)
#plt.figure()
#plt.plot(freqaxis,Spectrum)
#plt.gca().invert_xaxis()
#plt.show()  



#New Method===========================================
temptime = time.time()
RhoProp = Tinv @ RhoZero @ T
RhoProp =  np.tril(RhoProp,1)

DetectProp = np.real(Tinv @ Detect @ T)

Intensities = []
Frequencies = []
numtrans = 0 

AllInts = np.real(DetectProp * RhoProp)
for iii in range(RhoProp.shape[0]):
    for jjj in range(RhoProp.shape[0]):
        if abs(RhoProp[iii,jjj]) > 1e-9:
            Intensities.append(AllInts[iii,jjj])
            Frequencies.append(Hdiag[iii] - Hdiag[jjj])
            numtrans += 1

print('Get intensities = ' + str(time.time() - temptime) + ' s')   
         
Spectrum2, Axis = np.histogram(Frequencies, process, (-sw/2.0, sw/2.0) , weights = Intensities)
Fid = np.fft.ifft(np.fft.ifftshift(np.conj(scipy.signal.hilbert(Spectrum2))))
TimeAxis = np.linspace(0,process-1,process) * dw
Fid = Fid * np.exp(-TimeAxis * lb)

Spectrum2 = np.real(np.fft.fftshift(np.fft.fft(Fid)))
Spectrum2 = Spectrum2 / np.max(Spectrum2)

Axis = (Axis[1:] + 0.5 * (Axis[0] - Axis[1]))  / (42.576 * B0)


plt.figure()
plt.plot(Axis,Spectrum2)
#plt.plot(freqaxis,Spectrum)
plt.gca().invert_xaxis()
plt.show()  

print(numtrans)
print(((RhoProp.shape[0]**2)/2 -RhoProp.shape[0]) )
print(numtrans/((RhoProp.shape[0]**2)/2 -RhoProp.shape[0]) *100 )








