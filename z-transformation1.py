#%%
import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import glob



## access to pictures

# path to this Verzeichnis
current_directory = os.path.dirname(__file__)

# path to bilder --> training
bilder_ordner = os.path.join(current_directory, 'Bilder', 'Training')

# list of all data names in training
allpictures = os.listdir(bilder_ordner)
del allpictures[0] # bc first entry in pictures is readme



## pixelvalues of pictures in list

# empty list for pixelvalues of pictures
pixelwerte_liste = []

# loop to all the pictures
for sglpicture in allpictures:
     bildpfad = os.path.join(bilder_ordner, sglpicture)
    
     # open picture
     image = Image.open(bildpfad)
     # print(image)

       
     # convert it into numpy & pandas
     np_image = np.array(image)
     df = pd.DataFrame(data=np_image) 
                 
     # add pixelvalues to list
     pixelwerte_liste.append(df)


     
## list into vector into matrix & transponieren

erste_zeile = pixelwerte_liste[0].values.flatten()
pixel_matrix = np.expand_dims(erste_zeile, axis=0) 
for x in range(1, 120):
    pixelvalues = pixelwerte_liste[x]
    vector = pixelvalues.values.flatten()  # 1x77760 vector with pixelvalues
    vector = np.expand_dims(vector, axis=0)
    pixel_matrix = np.concatenate((pixel_matrix, vector), axis=0)

#print("pixel_matrix:", pixel_matrix)
pandas_pixel_matrix = pd.DataFrame(data=pixel_matrix)

pixel_matrix_t = np.transpose(pixel_matrix)
#print(pixel_matrix_t)



## histogram of intensitivalues

# pixel_matrix_t = pixel_matrix_t.flatten() # without it colourful histogram, but dont know what colours mean but better values on y-axis

plt.hist(pixel_matrix_t)
plt.title("Frequency of intensity values before z-transformation")
plt.xlabel("intensity values")
plt.ylabel("frequency")
plt.show()
pandas_pixel_matrix_t = pd.DataFrame(data=pixel_matrix_t)



## z-transformation preparation

mean = np.mean(pixel_matrix_t, axis=1)
sd = np.nanstd(pixel_matrix_t, axis=1)

sd[sd == 0] = 1 # to avoid dividing by 0
#nicht gut, das einfach auf 1 zu setzten, stattdessen bei 0 die rausnehmen  -> deshalb nicht so genau
   


## Z-Transformation
pixel_matrix_z = (pixel_matrix_t - mean[:, np.newaxis]) / sd[:, np.newaxis]

pandas_pixel_matrix_z = pd.DataFrame(data=pixel_matrix_z)

# print(pandas_pixel_matrix_z)

plt.hist(pixel_matrix_z, bins=50)
plt.title("Histogram of Z-transformed Data")
plt.xlabel("Z-transformed Values")
plt.ylabel("Frequency")
plt.show()


# %%
