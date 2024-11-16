import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

#define constants
H = np.array([[1,10],[10,2]])
initialstate = np.array([[1/np.sqrt(2)],[1/np.sqrt(2)]])
totaltime = 10
deltat = 0.01
#create lists of data points
t=[]
psi = []
alpha = []
beta = []
#initialize time
time = 0

#generate psi at each time step
while time <= totaltime:
  newstate = np.matmul(expm(-H*time),initialstate)
  newstate = newstate/np.linalg.norm(newstate)
  psi.append(newstate)
  t.append(time)
  time += deltat

#last calculation
t2 = np.matmul(expm(-H*time),initialstate)
t1 = np.matmul(expm(-H*(time-deltat)),initialstate)
print("E0 =",(-1/(time-(time-deltat)))*np.log(np.linalg.norm(t2)/np.linalg.
↪norm(t1)))

#extract alpha and beta from psi
for i in range(len(psi)):
  alpha.append(np.real(psi[i][0][0]*np.conjugate(psi[i][0][0])))
  beta.append(np.real(psi[i][1][0]*np.conjugate(psi[i][1][0])))

#make plots
aplot = plt.plot(t,alpha,label="alpha")
bplot = plt.plot(t,beta,label="beta")
plt.legend()
plt.show()
plt.show()
