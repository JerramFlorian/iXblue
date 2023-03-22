import numpy as np

# Charger le fichier texte en utilisant loadtxt
data = np.loadtxt("./Documents/iXblue/log_cali/temp_D.csv", delimiter=";", skiprows=1)

print(data)