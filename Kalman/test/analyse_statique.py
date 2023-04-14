import qrunch
import os
from roblib import *
from matplotlib.ticker import ScalarFormatter



#Importing the data (lat, lon, heading)
file_path = os.path.dirname(os.path.abspath(__file__)) + "\..\log"
gnss = np.load(file_path + "\gnss_data_qrunch_without_nmea.npz")
lat, lon = gnss["lat"], gnss["lon"]
print(lat, lon)
lat_m, lon_m = gnss["lat_m"], gnss["lon_m"]
print(lat_m, lon_m)
cap = gnss["heading"]
cov_latlat, cov_lonlon, cov_latlon, cov_hh, cov_pp, cov_hp = gnss["cov_latlat"], gnss["cov_lonlon"], gnss["cov_latlon"], gnss["cov_hh"], gnss["cov_pp"], gnss["cov_hp"]
innov_norm = np.load(file_path + "\innovation_normalisee.npz")["innovation_normalisee"]
DE, DN, DU = gnss["DE"], gnss["DN"], gnss["DU"] #-6.942984510888366 -2.435351410746334 -0.03586032939797576


#Converting into log_10 function
lg = lambda x : np.log10(x)


#Storing the data
N = 7
T, data, std = [[] for i in range(N)], [[] for i in range(N)], [[] for i in range(N)]
T[0], data[0], std[0] = qrunch.allan_deviation(lat)
T[1], data[1], std[1] = qrunch.allan_deviation(lon)
T[2], data[2], std[2] = qrunch.allan_deviation(DN)
T[3], data[3], std[3] = qrunch.allan_deviation(DE)
T[4], data[4], std[4] = qrunch.allan_deviation(cap)
T[5], data[5], std[5] = qrunch.allan_deviation(lat_m)
T[6], data[6], std[6] = qrunch.allan_deviation(lon_m)

# T = [t/3600 for t in T]


#Ploting the Allan deviation
fig, axs = plt.subplots(2, int(round(N/2+0.1)))
fig.suptitle(f"Déviation d'Allan")
titles = ["Lat [rad]", "Lon [rad]", "Delta North [m]", "Delta Est [m]", "heading [rad]", "Lat [m]", "Lon [m]"]
for i in range(2):
    for j in range(int(round(N/2+0.1))):
        ax = axs[i,j]
        quotient, remainder = divmod(j, int(N/2))
        index = quotient * int(N/2) + remainder + (N//2 * i)
        ax.loglog(T[index], data[index], label="allan")
        ax.loglog(T[index], std[index], label="écart-type")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(f"{titles[index]}")
        ax.legend()
# plt.show()


#Ploting the Allan deviation
fig, axs = plt.subplots(2, int(round(N/2+0.1)))
fig.suptitle(f"Déviation d'Allan")
titles = ["Lat [rad]", "Lon [rad]", "Delta North [m]", "Delta Est [m]", "heading [rad]", "Lat [m]", "Lon [m]"]
t0_rw, tf_rw = [0, 0, 4, 4, 4, 0, 0], [np.where(T[0]==1033)[0][0], np.where(T[0]==1033)[0][0], np.where(T[0]==210)[0][0], np.where(T[0]==210)[0][0], np.where(T[0]==210)[0][0], np.where(T[0]==1033)[0][0], np.where(T[0]==1033)[0][0]]
# t0_rw, tf_rw = [0, 0, 4, 4, 4, 4], [np.where(T[0]==1081)[0][0], np.where(T[0]==1081)[0][0], np.where(T[0]==208)[0][0], np.where(T[0]==208)[0][0], np.where(T[0]==208)[0][0], np.where(T[0]==208)[0][0]]
B = ["rw", "rw", "bc", "bc", "bc", "rw", "rw"]
sig = [np.sqrt(3)*10**(-8.125), np.sqrt(3)*10**(-7.731), 0.0825, 0.045, 0.03, np.sqrt(3)*10**(-2.931), np.sqrt(3)*10**(-2.568)]
# sig = [np.sqrt(3)*10**(-8.9), np.sqrt(3)*10**(-8.9), 0.07, 0.045, 0.025, 0.025]
for i in range(2):
    for j in range(int(round(N/2+0.1))):
        ax = axs[i,j]
        quotient, remainder = divmod(j, int(N/2))
        index = quotient * int(N/2) + remainder + (N//2 * i)
        a, b = np.polyfit(lg(T[index][t0_rw[index]:tf_rw[index]]), lg(data[index][t0_rw[index]:tf_rw[index]]), 1)
        ax.loglog(T[index], data[index], label=f"allan : {'%.4g'%a}t + {'%.4g'%b}")
        y_rw = sig[index]/np.sqrt(3)*np.sqrt(T[index])
        y_bb = sig[index]/np.sqrt(T[index])
        if B[index] == "rw":
            a_th, b_th = np.polyfit(lg(T[index]), lg(y_rw), 1)
            ax.loglog(T[index], y_rw, label=f"{B[index]} : {'%.4g'%a_th}t + {'%.4g'%b_th}")           
        if B[index] == "bc":
            tau = [[1, 10, 100, 4000], [1, 10, 125, 1500, 10000], [1, 10, 100, 3750], [1, 10, 100, 3750]]
            cpt = 1
            for t in tau[index-2]:
                exec(f"y2_bc_{index-1}_{cpt} = 2*t*sig[index]**2/T[index]*(1-t/(2*T[index])*(3-4*np.exp(-T[index]/t)+np.exp(-2*T[index]/t)))")
                cpt += 1
            ax.loglog(T[index], np.sqrt(sum(eval(f'y2_bc_{index-1}_{i}**2') for i in range(1, len(tau[index-2])+1))), label=f"{B[index]} : {tau[index-2]}")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(f"{titles[index]}")
        ax.legend()
plt.show()


# #Ploting the Allan deviation
# fig, axs = plt.subplots(2, int(round(N/2+0.1)))
# fig.suptitle(f"Déviation d'Allan")
# titles = ["Lat [m]", "Lon [m]", "Delta North [m]", "Delta Est [m]", "heading [rad]", "heading [°]"]
# t0_bb, tf_bb = [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]
# t0_rw, tf_rw = [0, 0, 4, 4, 4, 4], [np.where(T[0]==1033)[0][0], np.where(T[0]==1033)[0][0], np.where(T[0]==210)[0][0], np.where(T[0]==210)[0][0], np.where(T[0]==210)[0][0], np.where(T[0]==210)[0][0]]
# V = [np.mean(cov_latlat), np.mean(cov_lonlon), np.std(DN), np.std(DE), np.mean(cov_hh), np.std(head)]
# B = ["rw", "rw", "de+bb", "de+bb", "de+bb", "de+bb"]
# for i in range(2):
#     for j in range(int(round(N/2+0.1))):
#         ax = axs[i,j]
#         quotient, remainder = divmod(j, int(N/2))
#         index = quotient * int(N/2) + remainder + (N//2 * i)
#         ax.loglog(T[index], data[index], label="allan")

#         if t0_bb[index] != tf_bb[index]:
#             a_bb, b_bb = np.polyfit(lg(T[index][t0_bb[index]:tf_bb[index]]), lg(data[index][t0_bb[index]:tf_bb[index]]), 1)
#             a_bb_th, b_bb_th = [-1/2, lg(np.std(data[index][t0_bb[index]:tf_bb[index]]))]
#             ax.loglog(T[index][t0_bb[index]:tf_bb[index]], T[index][t0_bb[index]:tf_bb[index]]**a_bb/10**(-b_bb), label=f"bb : {'%.4g'%a_bb}t {'%.4g'%b_bb}")
#             ax.loglog(T[index][t0_bb[index]:tf_bb[index]], T[index][t0_bb[index]:tf_bb[index]]**a_bb_th/10**(-b_bb_th), label=f"bb : {'%.4g'%a_bb_th}t {'%.4g'%b_bb_th}")

#         if t0_rw[index] != tf_rw[index]:
#             a_rw, b_rw = np.polyfit(lg(T[index][t0_rw[index]:tf_rw[index]]), lg(data[index][t0_rw[index]:tf_rw[index]]), 1)
#             if B[index] == "rw":
#                 a_rw_th, b_rw_th = [1/2, lg(V[index]/np.sqrt(3))]
#             if B[index] == "de+bb":
#                 a_rw_th, b_rw_th = [1/4, 0.5*lg(V[index]**2/np.sqrt(2))]
#             ax.loglog(T[index][t0_rw[index]:tf_rw[index]], T[index][t0_rw[index]:tf_rw[index]]**a_rw/10**(-b_rw), label=f"{B[index]} : {'%.4g'%a_rw}t {'%.4g'%b_rw}")
#             ax.loglog(T[index][t0_rw[index]:tf_rw[index]], T[index][t0_rw[index]:tf_rw[index]]**a_rw_th/10**(-b_rw_th), label=f"{B[index]} : {'%.4g'%a_rw_th}t {'%.4g'%b_rw_th}")

#         ax.set_xlabel("Time [s]")
#         ax.set_ylabel(f"{titles[index]}")
#         ax.legend()

#         #Limiter les chiffres significatifs
#         ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
#         ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
# plt.show()


if __name__ == "__main__":
    save_data = True
    if save_data == True:
        np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)) + "\..\log", "deviation_allan.npz"), T=T[0], dev_lat=data[5], dev_lon=data[6], dev_cap=data[4])