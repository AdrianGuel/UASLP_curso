import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def Scalexy(data, x0, x1, y0, y1):
  m = (y1 - y0)/(x1 - x0)
  b = y1 - m*x1
  y = m*data + b
  return y

'''
Controlador neuronal para una planta de segundo orden

Planta:

               18
  G(S) = ---------------
         s^2 + 2.4 s + 9

Utilizando el comando "c2d" en Matlab: GZ = c2d(GS,0.1,'zoh')


           0.0826 z + 0.07623
G(Z) =   ----------------------
         z^2 - 1.707 z + 0.7866

  
'''


e = 0
e1 = 0
e2 = 0
e3 = 0

y = 0
y1 = 0
y2 = 0

u = 0
u1 = 0
u2 = 0

s1 = 0
s2 = 0
s3 = 0


#Inicialización aleatoria de los pesos de la capa de entrada
w11 = np.random.random()
w12 = np.random.random()
w13 = np.random.random()
w21 = np.random.random()
w22 = np.random.random()
w23 = np.random.random()
w31 = np.random.random()
w32 = np.random.random()
w33 = np.random.random()

#Inicialización aleatoria de los pesos de la capa oculta
v1 = np.random.random()
v2 = np.random.random()
v3 = np.random.random()

#Términos para el factor de aprendizaje dinámico
alpha = 0.25
eta = 0.4


it = 0
lim = 500

#Set-Point
sp = 1

#Signo de la planta
signD = 1

#Vectores auxiliares para graficar
tk = []
yk = []
spk = []
uk = []

while it < lim:
  #Cálculo de las sumatorias en las neuronas de entrada:
  s1=e1*w11+e2*w21+e3*w31
  s2=e1*w12+e2*w22+e3*w32
  s3=e1*w13+e2*w23+e3*w33
   
  #Cálculo de las salidas de las neuronas de entrada:
  h1=1/(1+np.exp(-s1))
  h2=1/(1+np.exp(-s2))
  h3=1/(1+np.exp(-s3))
   
  #Cálculo de la sumatoria en la neurona de salida:
  r=h1*v1+h2*v2+h3*v3
  
  #Cálculo de la salida de la red
  u=1/(1+np.exp(-r))

  #Escalamiento de la señal de control, descomentar en caso necesario             
  #u=Scalexy(u,0,1,0,100)

  #Planta:
  y = 0.0826*u1 + 0.07623*u2 + 1.707*y1 - 0.7866*y2

  #Cálculo del error del sistema
  e = sp - y
   
  #Se implementó un factor de aprendizaje dinámico.
  ra = eta + alpha*abs(e)

  delta1 = e*(u*(1-u))
   
  delta21=delta1*v1*(h1*(1-h1))
  delta22=delta1*v2*(h2*(1-h2))
  delta23=delta1*v3*(h3*(1-h3))
  
  #Ecuaciones de optimización para los pesos:
  v1 = v1 + ra*delta1*signD*h1
  v2 = v2 + ra*delta1*signD*h2
  v3 = v3 + ra*delta1*signD*h3
  
  w11 = w11 + ra*delta21*signD*e1
  w12 = w12 + ra*delta22*signD*e1
  w13 = w13 + ra*delta23*signD*e1
  
  w21 = w21 + ra*delta21*signD*e2
  w22 = w22 + ra*delta22*signD*e2
  w23 = w23 + ra*delta23*signD*e2
  
  w31 = w31 + ra*delta21*signD*e3
  w32 = w32 + ra*delta22*signD*e3
  w33 = w33 + ra*delta23*signD*e3

   
  if it%2 == 0:
    print(sp, y, e)
    tk.append(it)
    yk.append(y)
    spk.append(sp)
    uk.append(u)
  

  
  e3 = e2
  e1 = e

  y2 = y1
  y1 = y

  u2 = u1
  u1 = u

  
  it += 1

  if it%250 == 0:
    sp = sp - 0.5

import seaborn as sns
sns.set_theme(style="darkgrid")

plt.plot(tk,spk, label = "Set-point", linestyle = "--", c = "red")
plt.plot(tk,yk, label = "PV", c = "darkblue")
plt.xlabel("Épocas")
plt.ylabel("PV")
plt.legend()
plt.show()
