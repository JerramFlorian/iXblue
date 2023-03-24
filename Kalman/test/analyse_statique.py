import qrunch
import os
from roblib import *
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter


#Importing the data (lat, lon, heading)
file_path = os.path.dirname(os.path.abspath(__file__)) + "\..\log"
gnss = np.load(file_path + "\gnss_data_qrunch_with_nmea.npz")
lat, lon = gnss["lat"], gnss["lon"]
cap = gnss["heading"]
cov_latlat, cov_lonlon, cov_latlon, cov_hh, cov_pp, cov_hp = gnss["cov_latlat"], gnss["cov_lonlon"], gnss["cov_latlon"], gnss["cov_hh"], gnss["cov_pp"], gnss["cov_hp"]
innov_norm = np.load(file_path + "\innovation_normalisee.npz")["innovation_normalisee"]


#Storing the data
N = 5
T, data, std = [[] for i in range(N)], [[] for i in range(N)], [[] for i in range(N)]
T[0], data[0], std[0] = qrunch.allan_deviation(lat)
T[1], data[1], std[1] = qrunch.allan_deviation(lon)
T[2], data[2], std[2] = qrunch.allan_deviation(cap)
T[3], data[3], std[3] = qrunch.allan_deviation(cov_hh)
# T[4], data[4], std[4] = qrunch.allan_deviation(innov_norm[:, 0])
# T[5], data[5], std[5] = qrunch.allan_deviation(innov_norm[:, 1])
T[4], data[4], std[4] = qrunch.allan_deviation(innov_norm[:, 2])

# T = [t/3600 for t in T]


#Ploting the Allan deviation
fig, axs = plt.subplots(2, N)
fig.suptitle(f"Déviation d'Allan")
data_list = [data, std]
labels = ["Allan", "Ecart-type"]
for i in range(2):
    for j in range(N):
        ax = axs[i,j]
        ax.loglog(T[j], data_list[i][j])
        ax.set_xlabel("Time [h]")
        ax.set_ylabel(f"{labels[i]}")

        #Limiter les chiffres significatifs
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
# plt.show()


#Ploting the Allan deviation
fig, axs = plt.subplots(2, int(round(N/2+0.1)))
fig.suptitle(f"Déviation d'Allan")

for i in range(2):
    for j in range(int(round(N/2+0.1))):
        ax = axs[i,j]
        quotient, remainder = divmod(j, int(N/2))
        index = quotient * int(N/2) + remainder + (N//2 * i)
        ax.loglog(T[index], data[index], label="Allan")
        ax.loglog(T[index], std[index], label="Ecart-type")
        ax.set_xlabel("Time [h]")
        ax.legend()

        #Limiter les chiffres significatifs
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
# plt.show()


#Ploting the Allan deviation
fig, axs = plt.subplots(1, 2)
fig.suptitle(f"Déviation d'Allan")
titles = ["Allan", "Ecart-type"]
data_list = [data, std]
labels = ['lat', 'lon', 'cap', 'cov_hh', 'innov_norm_cap']
for i, ax in enumerate(axs):
    ax.set_title(titles[i])
    ax.set_xlabel('Time [h]')
    for j, l in enumerate(labels):
        ax.loglog(T[j], data_list[i][j], label=f"{l}")
    ax.legend()
plt.show()


# qrunch.qallan()