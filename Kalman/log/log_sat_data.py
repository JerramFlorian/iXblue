import numpy as np
from numpy import cos, sin
import os


#Importing the data
file_path = os.path.dirname(os.path.abspath(__file__)) + "\sat_data"
satvis_path = file_path + '\long_acquisition_sat_vis.txt'
cpd_path = file_path + '\exail_test_station.sbf_SBF_MeasEpoch2.txt'

satvis = np.genfromtxt(satvis_path, delimiter = ',', dtype = 'str', skip_footer = 3)
cpd_data = np.genfromtxt(cpd_path, delimiter = ',', dtype = 'str', skip_footer = 3)


#Extracting the number of satellites
def sat_number(X, j):
    try:
        nb_sat = np.float64(X[:, j])
    except:
        print("Warning : nb_sat data weren't rightly saved !")
        cpt = 0
        nb_sat = []
        for i in range(len(X[:, j])):
            if X[i, j] != '':
                nb_sat.append(float(X[i, j]))
            else:
                cpt += 1
        print("Number of erroneous data : ", cpt)
    return(nb_sat)  

#Extracting elevation data
def elevation(X, j):
    try:
        elev = np.float64(X[:, j])
    except:
        print("Warning : elevation data weren't rightly saved !")
        cpt = 0
        elev = []
        for i in range(len(X[:, j])):
            if X[i, j] != '':
                elev.append(float(X[i, j]))
            else:
                cpt += 1
        print("Number of erroneous data : ", cpt)
    return(elev)

#Extracting azimuth data
def azimuth(X, j):
    try:
        az = np.float64(X[:, j])
    except:
        print("Warning : azimuth data weren't rightly saved !")
        cpt = 0
        az = []
        for i in range(len(X[:, j])):
            if X[i, j] != '':
                az.append(float(X[i, j]))
            else:
                cpt += 1
        print("Number of erroneous data : ", cpt)
    return(az)

#Extracting carrier phase differences
def CPD(X):
    cp = [np.float64(X[0, -1]), np.float64(X[2, -1])]
    cpd = (cp[1]-cp[0])/np.float64(X[0, -6])
    return(cpd) 



if __name__ == "__main__":
    print("\n----- Extracting the satellite vis data -----")
    elev_satvis = elevation(satvis, 16) ; azim_satvis = azimuth(satvis, 14)
    nb_sat = sat_number(satvis, 10)
    np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sat_data_vis.npz"), elev=elev_satvis, azim=azim_satvis, nb_sat=nb_sat)
    print("----- Saving the satellite vis data -----\n")

    print("----- Extracting the measure data -----")
    L = np.float64(cpd_data[:, -5]) ; f = np.float64(cpd_data[:, -6])
    cpd_mean = CPD(cpd_data)
    np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "measure.npz"), L=L, f=f, Dph=cpd_mean)
    print("----- Saving the emasure data -----\n")
