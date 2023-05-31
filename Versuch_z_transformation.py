from matplotlib import transforms
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import os
import pandas as pd

#Pfad zum Ordner
Ordner = "Bilder\Training"
#Liste erstellen, in der die Bilder dann aufeinanderfolgende Zahlen sind für den Loop
Pixelwerte = []
for dateiname in os.listdir(Ordner): #Dateien aus Ordner aufrufen
    if dateiname.endswith("gif"): #nur Bilder aus Ordner aufrufen
        bildpfad = os.path.join(Ordner, dateiname)
        bild = imageio.imread(bildpfad) 
        bild_pixelwerte = bild.flatten() #Pixelwerte in flacher Liste anordnen, also als Vektor mit der Dimension 1xn
        Pixelwerte.append(bild_pixelwerte) #neue Pixelwerte der Liste hinzufügen
Matrix = np.column_stack(Pixelwerte) #fügt die flachen Listen zu der Matrix hinzu
print(Matrix) #Matrix mit den Pixelwerten aller Bilder, für jedes Bild eine Spalte. Das heißt, die Z-Transformation muss jeweils pro Zeile durchgeführt werden.
print(Matrix.shape) #nachschauen, ob tatsächlich 77760 Zeilen (Anzahl der Pixel) und 120 Spalten (Anzahl der Bilder vorhanden sind)
        
#graphisch visualisieren für vorher-nachher

hist, edges = np.histogram(Matrix)
#Histogramm plotten
plt.bar(edges[:-1], hist, width=0.9)
plt.xlabel("Pixelwerte")
plt.ylabel("Häufigkeit")
plt.title("Häufigkeit der Pixelwerte vor der Transformation")
plt.show()

#Programm für die Z-Transformation schreiben:
#definition
def z_transformation(Matrix):
    transformed_matrix = []
    #dafür muss ich aus der Matrix jeweils die Werte aus einer Zeile nehmen und über diese 120 Werte normalisieren
    for row in Matrix:
        sd = np.std(row) #Standardabweichung jeder Zeile berechnen
        if sd == 0:
            tranformed_row = row - mean
        else:
            mean = np.mean(row) #Mittelwert jeder Zeile berechnen
            transformed_row = (row - mean) / sd #Z-Transformation auf jede Zeile anwenden
        transformed_matrix.append(transformed_row) #die transformierten Zeilen in die transformed Matrix aufnehmen
    return transformed_matrix
neue_matrix = Matrix
transformierte_Matrix = z_transformation(neue_matrix) #Z-Transformation auf die ganze Matrix zeilenweise anwenden anwenden
#transformierte Werte ausgeben
#überprüfen, ob die Werte normalisiert sind mit graphischer Visualisierung

# Alle transformierten Werte in einen flachen Array umwandeln, damit ich nicht 120 Histogramme ausgegeben bekomme:
transformierte_werte = np.concatenate(transformierte_Matrix)
#transformierte_werte = transformierte_werte[~np.isnan(transformierte_werte)] #NaN-Werte (not a number) können auftreten, wenn sd=0, versucher ich noch zu beheben. Aber was genau sollen wir dann tun?
#Histogramm für die transformierten Werte plotten
hist, edges = np.histogram(transformierte_werte)
plt.bar(edges[:-1], hist, width=0.9)
plt.xlabel("Pixelwerte")
plt.ylabel("Häufigkeit")
plt.title("Häufigkeit der Pixelwerte nach der Transformation")
plt.show()
Mittelwert = np.mean(transformierte_werte)
Standardabweichung = np.std(transformierte_werte)
print(Mittelwert)
print(Standardabweichung)
