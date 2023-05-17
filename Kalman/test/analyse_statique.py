import qrunch
import os
from roblib import *



#Importing the data (lat, lon, heading)
file_path = os.path.dirname(os.path.abspath(__file__)) + "\..\log"
gnss = np.load(file_path + "\gnss_data_qrunch_without_nmea.npz")
# lat, lon = gnss["lat"], gnss["lon"] # radian
lat, lon = gnss["lat_m"], gnss["lon_m"] # m
print(lat, lon)
cap = gnss["heading"]
cov_latlat, cov_lonlon, cov_latlon, cov_hh, cov_pp, cov_hp = gnss["cov_latlat"], gnss["cov_lonlon"], gnss["cov_latlon"], gnss["cov_hh"], gnss["cov_pp"], gnss["cov_hp"]
innov_norm = np.load(file_path + "\innovation_normalisee.npz")["innovation_normalisee"]
innov = np.load(file_path + "\innovation_normalisee.npz")["innovation"]
DE, DN, DU = gnss["DE"], gnss["DN"], gnss["DU"] #-6.942984510888366 -2.435351410746334 -0.03586032939797576


#Converting into log_10 function
lg = lambda x : np.log10(x)


#Storing the data
N = 8
T, data, std = [[] for i in range(N)], [[] for i in range(N)], [[] for i in range(N)]
T[0], data[0], std[0] = qrunch.allan_deviation(lat)
T[1], data[1], std[1] = qrunch.allan_deviation(lon)
T[2], data[2], std[2] = qrunch.allan_deviation(DN)
T[3], data[3], std[3] = qrunch.allan_deviation(DE)
T[4], data[4], std[4] = qrunch.allan_deviation(cap)
T[5], data[5], std[5] = qrunch.allan_deviation(innov_norm[:, 0])
T[6], data[6], std[6] = qrunch.allan_deviation(innov_norm[:, 1])
T[7], data[7], std[7] = qrunch.allan_deviation(innov_norm[:, 2])


#Ploting the Allan deviation
fig, axs = plt.subplots(2, int(round(N/2+0.1)))
fig.suptitle(f"Déviation d'Allan")
titles = ["Lat [m]", "Lon [m]", "Delta North [m]", "Delta Est [m]", "heading [rad]", "innov_norm Lon [m]", "innov_norm Lat [m]", "innov_norm Cap [rad]"]
for i in range(2):
    for j in range(int(round(N/2+0.1))):
        ax = axs[i,j]
        quotient, remainder = divmod(j, int(N/2))
        index = quotient * int(N/2) + remainder + (N//2 * i)
        ax.loglog(T[index]/60, data[index], label="allan")
        ax.loglog(T[index]/60, std[index], label="écart-type")
        ax.set_xlabel("Time [min]")
        ax.set_ylabel(f"{titles[index]}")
        ax.legend()
# plt.show()


#Ploting the Allan deviation
fig, axs = plt.subplots(2, int(round(N/2+0.1)))
fig.suptitle(f"Déviation d'Allan")
titles = ["Lat [m]", "Lon [m]", "Delta North [m]", "Delta Est [m]", "heading [rad]", "innov_norm Lon [m]", "innov_norm Lat [m]", "innov_norm Cap [rad]"]
B = ["rw", "rw", "bc", "bc", "bc", "bb", "bb", "bb"]
sig = [0.002, 0.0045, 0.004, 0.0015, 0.0005, 1, 1, 1]
for i in range(2):
    for j in range(int(round(N/2+0.1))):
        ax = axs[i,j]
        quotient, remainder = divmod(j, int(N/2))
        index = quotient * int(N/2) + remainder + (N//2 * i)
        a, b = np.polyfit(lg(T[index]), lg(data[index]), 1)
        ax.loglog(T[index]/60, data[index], label=f"variance d'Allan")
        y_rw = sig[index]/np.sqrt(3)*np.sqrt(T[index])
        y_bb = sig[index]/np.sqrt(T[index])
        if B[index] == "rw":
            a_th, b_th = np.polyfit(lg(T[index]), lg(y_rw), 1)
            ax.loglog(T[index]/60, y_rw, label=f"{B[index]} : {'%.4g'%a_th}t + {'%.4g'%b_th}")    
        if B[index] == "bb":
            a_th, b_th = np.polyfit(lg(T[index]), lg(y_bb), 1)
            ax.loglog(T[index]/60, y_bb, label=f"{B[index]} : {'%.4g'%a_th}t + {'%.4g'%b_th}")        
        if B[index] == "bc":
            tau = [[0.025, 200, 4000], [0.1, 75, 200, 2500, 3500], [0.025, 200, 4000]]
            cpt = 1
            for t in tau[index-2]:
                exec(f"y2_bc_{index-1}_{cpt} = 2*t*sig[index]**2/T[index]*(1-t/(2*T[index])*(3-4*np.exp(-T[index]/t)+np.exp(-2*T[index]/t)))")
                cpt += 1
            ax.loglog(T[index]/60, np.sqrt(sum(eval(f'y2_bc_{index-1}_{i}') for i in range(1, len(tau[index-2])+1))), label=f"{B[index]} : {tau[index-2]}")
        ax.set_xlabel("Time [min]")
        ax.set_ylabel(f"{titles[index]}")
        ax.legend()
# plt.show()


#Ploting the Allan deviation
fig, axs = plt.subplots(3)
fig.suptitle(f"Déviation d'Allan")
titles = ["innov_norm Lon [m]", "innov_norm Lat [m]", "innov_norm Cap [rad]"]
B = ["bb", "bb", "bb"]
sig = [1, 1, 1]
for i in range(3):
    ax = axs[i]
    i-=3
    a, b = np.polyfit(lg(T[i]), lg(data[i]), 1)
    ax.loglog(T[i]/60, data[i], label=f"variance d'Allan : {'%.4g'%a}t + {'%.4g'%b}")
    y_rw = sig[i]/np.sqrt(3)*T[i]
    y_bb = sig[i]/np.sqrt(T[i])
    if B[i] == "rw":
        a_th, b_th = np.polyfit(lg(T[i]), lg(y_rw), 1)
        ax.loglog(T[i]/60, y_rw, label=f"{B[i]} : {'%.4g'%a_th}t + {'%.4g'%b_th}")    
    if B[i] == "bb":
        a_th, b_th = np.polyfit(lg(T[i]), lg(y_bb), 1)
        ax.loglog(T[i]/60, y_bb, label=f"{B[i]} : {'%.4g'%a_th}t + {'%.4g'%b_th}")        
    if B[i] == "bc":
        tau = [[0.025, 200, 4000], [0.1, 75, 200, 2500, 3500], [0.025, 200, 4000]]
        cpt = 1
        for t in tau[i-2]:
            exec(f"y2_bc_{i+3}_{cpt} = 2*t*sig[i]**2/T[i]*(1-t/(2*T[i])*(3-4*np.exp(-T[i]/t)+np.exp(-2*T[i]/t)))")
            cpt += 1
        ax.loglog(T[j]/60, np.sqrt(sum(eval(f'y2_bc_{j+3}_{j}') for j in range(1, len(tau[i-2])+1))), label=f"{B[j]} : {tau[j-2]}")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel(f"{titles[i]}")
    ax.legend()
plt.show()


if __name__ == "__main__":
    save_data = True
    if save_data == True:
        np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)) + "\..\log", "deviation_allan.npz"), T=T[0], dev_lat=data[0], dev_lon=data[1], dev_cap=data[4])