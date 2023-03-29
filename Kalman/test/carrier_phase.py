import numpy as np
from numpy import cos, sin
import os
from tqdm import tqdm


#Importing the data
file_path = os.path.dirname(os.path.abspath(__file__)) + "\..\log"
sat_vis = np.load(file_path + "\sat_data_vis.npz")
cpd = np.load(file_path + "\measure.npz")
dph = cpd["Dph"] ; L = cpd["L"][0];  f = cpd["f"][0]
azim_satvis = sat_vis["azim"] ; elev_satvis = sat_vis["elev"] ; nb_sat = sat_vis["nb_sat"]


#Gauss-Newton
def Gauss_Newton(X0):
    X = X0
    N_av = 0
    for i in tqdm(np.arange(0, len(nb_sat), 1)):
        try:
            N = int(nb_sat[N_av*i])
            # print('nb sat : ', N)
            az, el = azim_satvis[N_av*i:N_av*i+N], elev_satvis[N_av*i:N_av*i+N]
            # Dph = np.array([[CPD(cpd_data, 2*(N_av*i), 2*(N_av*i+N))] for _ in range(N)])
            Dph = np.array([[dph] for _ in range(N)])

            r = np.array([[0] for _ in range(N)])
            for i in range(N):
                r[i, 0] = Dph[i, 0] - D/L * (sin(az[i])*cos(el[i])*sin(X[0, 0])*cos(X[1, 0]) + cos(az[i])*cos(el[i])*cos(X[0, 0])*cos(X[1, 0]) + sin(az[i])*sin(X[1, 0]))

            J = D/L * np.ones((N, 2))
            for i in range(N):
                J[i][0] = cos(el[i])*cos(X[1, 0]) * (cos(az[i])*sin(X[0, 0]) - sin(az[i])*cos(X[0, 0]))
                J[i][1] = sin(az[i])*cos(el[i])*sin(X[0, 0])*sin(X[1, 0]) + cos(az[i])*cos(el[i])*cos(X[0, 0])*sin(X[1, 0]) - sin(az[i])*cos(X[1, 0])

            X -= np.linalg.inv(J.T@J)@J.T@r

            N_av = N
        except:
            # print("Fini")
            pass
    print("Dph : ", Dph[0])
    return X



if __name__ == "__main__":
    X0 = np.array([[250, -0.3]]).T
    D = np.sqrt((6.942)**2 + (2.435)**2 + (0.039)**2)
    X = Gauss_Newton(X0)
    print('X : ', X)