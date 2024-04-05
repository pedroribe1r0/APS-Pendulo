import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit
sns.set_theme()

def aps(t, a, w, b, p):
    return a*np.exp((-b*t)) * np.cos(w * t - p)

vid = cv2.VideoCapture("./Pendulo_vid.mov")
#detector para remover o fundo
detector = cv2.createBackgroundSubtractorKNN(history = 2000, dist2Threshold=500, detectShadows=False)

fps = vid.get(cv2.CAP_PROP_FPS)
#lista para armazenar os dados coletados
dados = []

t = 0

while 1:
    
    #interromper o programa
    key = cv2.waitKey(20)
    if key == 8: #delete
        break
    
    #pegar frame por frame
    working, frame = vid.read()
    #remove o fundo
    filter = detector.apply(frame)
    
    contorno, _ = cv2.findContours(filter, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for cont in contorno:
        area = cv2.contourArea(cont)
        if area > 200:
            x, y, z, h = cv2.boundingRect(cont)
            cv2.rectangle(frame, (x, y), (x+z, y+h),(0, 0, 255), 2)
            
            dados.append({"t": t, "x": (x + z/2)})
            break 
    
           
    
    cv2.imshow("FRM", frame)
    cv2.imshow("FLT", filter)
    
    
    t += 1.0/fps
    

vid.release()
cv2.destroyAllWindows()

#tratamento dos dados

df = pd.DataFrame(dados)

df = df[200:]

#converter para metros

df["x"] -= (np.max(df["x"]) + np.min(df["x"]))/2

df["x"] *= 2/(np.max(df["x"]) - np.min(df["x"]))

df["x"] *= (60/2)/100
#achar as variaveis da função
popt, _ = curve_fit(aps, df["t"], df["x"])

periodo = (2 * np.pi)/(popt[1])

print(periodo)

print(popt)

#calculo do fator de qualidade
fq = 2*np.pi/(1-np.exp(-2*popt[2]*periodo))

print(fq)
#plotar o grafico
sns.scatterplot(df, x="t", y="x")
plt.show()











