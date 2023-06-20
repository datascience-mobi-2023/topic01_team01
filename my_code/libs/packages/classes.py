from matplotlib import transforms
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn import decomposition
from PIL import Image

def z_transformation(Matrix):
    transformed_matrix = []
    #dafür muss ich aus der Matrix jeweils die Werte aus einer Zeile nehmen und über diese 120 Werte normalisieren
    for row in Matrix:
        sd = np.std(row) #Standardabweichung jeder Zeile berechnen
        if sd == 0:
            tranformed_row = row - mean #hier keine transformation!!!!
        else:
            mean = np.mean(row) #Mittelwert jeder Zeile berechnen
            transformed_row = (row - mean) / sd #Z-Transformation auf jede Zeile anwenden
        transformed_matrix.append(transformed_row) #die transformierten Zeilen in die transformed Matrix aufnehmen
    return transformed_matrix



n_row, n_col = 4,5 # adat to number of PCs
n_components = n_row * n_col
image_shape = (243, 320) 


def plot_gallery(title, images, n_col=n_col, n_row=n_row, cmap=plt.cm.gray):
    fig, axs = plt.subplots(
        nrows=n_row,
        ncols=n_col,
        figsize=(2.0 * n_col, 2.3 * n_row),
        facecolor="white",
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.02, hspace=0, wspace=0)
    fig.set_edgecolor("black")
    fig.suptitle(title, size=16)
    for ax, vec in zip(axs.flat, images):
        vmax = max(vec.max(), -vec.min())
        im = ax.imshow(
            vec.reshape(image_shape),
            cmap=cmap,
            interpolation="nearest",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.axis("off")

    fig.colorbar(im, ax=axs, orientation="horizontal", shrink=0.99, aspect=40, pad=0.01)
    plt.show()