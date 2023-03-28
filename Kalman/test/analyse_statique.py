import qrunch
import os
from roblib import *
from matplotlib.ticker import ScalarFormatter
# from sklearn.metrics import r2_score



#Importing the data (lat, lon, heading)
file_path = os.path.dirname(os.path.abspath(__file__)) + "\..\log"
gnss = np.load(file_path + "\gnss_data_qrunch_without_nmea.npz")
lat, lon = gnss["lat"], gnss["lon"]
cap = gnss["heading"]
cov_latlat, cov_lonlon, cov_latlon, cov_hh, cov_pp, cov_hp = gnss["cov_latlat"], gnss["cov_lonlon"], gnss["cov_latlon"], gnss["cov_hh"], gnss["cov_pp"], gnss["cov_hp"]
innov_norm = np.load(file_path + "\innovation_normalisee.npz")["innovation_normalisee"]


#Storing the data
N = 2
T, data, std = [[] for i in range(N)], [[] for i in range(N)], [[] for i in range(N)]
# T[0], data[0], std[0] = qrunch.allan_deviation(lat, 1)
# T[1], data[1], std[1] = qrunch.allan_deviation(lon, 1)
T[0], data[0], std[0] = qrunch.allan_deviation(cap, 1)
# T[3], data[3], std[3] = qrunch.allan_deviation(cov_hh, 1)
# T[4], data[4], std[4] = qrunch.allan_deviation(innov_norm[:, 0])
# T[5], data[5], std[5] = qrunch.allan_deviation(innov_norm[:, 1])
T[1], data[1], std[1] = qrunch.allan_deviation(innov_norm[:, 2], 1)
# T = [t/3600 for t in T]


# #Ploting the Allan deviation
# fig, axs = plt.subplots(2, N)
# fig.suptitle(f"Déviation d'Allan")
# data_list = [data, std]
# labels = ["Allan déviation", "Allan std"]
# units = ["[m]", "[m]", "[°]", "[°]"]
# for i in range(2):
#     for j in range(N):
#         ax = axs[i,j]
#         ax.loglog(T[j], data_list[i][j])
#         ax.set_xlabel("Time [s]")
#         ax.setylabel(f"Unit : {units[j]}")
#         ax.set_ylabel(f"{labels[i]}")

#         #Limiter les chiffres significatifs
#         ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
#         ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
# # plt.show()


#Linear regression function
def linear_regression(x, y) :
    coef = np.polyfit(x, y, 1)
    return coef


#Converting into log_10 function
lg = lambda x : np.log(x)/np.log(10)


# #Ploting the Allan deviation
# fig, axs = plt.subplots(2, int(round(N/2+0.1)))
# fig.suptitle(f"Déviation d'Allan")
# units = ["[m]", "[m]", "[°]", "[°]"]
# for i in range(2):
#     for j in range(int(round(N/2+0.1))):
#         ax = axs[i,j]
#         quotient, remainder = divmod(j, int(N/2))
#         index = quotient * int(N/2) + remainder + (N//2 * i)
#         a_bb, b_bb = linear_regression(log(T[index][:]), log(data[index][:]))
#         a_rw, b_rw = linear_regression(log(T[index][:]), log(data[index][:]))
#         a_bb_th, b_bb_th = [-1/2, np.std(data[index][:])]
#         a_rw_th, b_rw_th = [1/2, np.std(data[index][:])-log(np.sqrt(3))]
#         ax.loglog(T[index], data[index], label="Allan")
#         # ax.loglog(T[index], std[index], label="std")
#         ax.plot(log(T[index]), a_bb*log(data[index])+b_bb, label=f'bb : {a_bb}x+{b_bb}')
#         ax.plot(log(T[index]), a_rw*log(data[index])+b_rw, label=f'rw : {a_rw}x+{b_rw}')
#         ax.plot(log(T[index]), a_bb_th*log(data[index])+b_bb_th, label=f'bb : {a_bb_th}x+{b_bb_th}')
#         ax.plot(log(T[index]), a_rw_th*log(data[index])+b_rw_th, label=f'rw : {a_rw_th}x+{b_rw_th}')
#         ax.set_xlabel("Time [s]")
#         ax.set_ylabel(f"Units : {units[j]}")
#         ax.legend()

#         #Limiter les chiffres significatifs
#         ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
#         ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
# # plt.show()


#Ploting the Allan deviation
fig, axs = plt.subplots(2, int(round(N/2+0.1)))
fig.suptitle(f"Déviation d'Allan")
units = ["[°]", "[°]"]
t0_bb, tf_bb = 0, 5
t0_rw, tf_rw = 4, np.where(T[0]==334)[0][0]
for i in range(2):
    for j in range(int(round(N/2+0.1))):
        ax = axs[i]
        quotient, remainder = divmod(j, int(N/2))
        index = quotient * int(N/2) + remainder + (N//2 * i)
        a_bb, b_bb = linear_regression(lg(T[index][t0_bb:tf_bb]), lg(data[index][t0_bb:tf_bb]))
        a_rw, b_rw = linear_regression(lg(T[index][t0_rw:tf_rw]), lg(data[index][t0_rw:tf_rw]))
        a_bb_th, b_bb_th = [-1/2, lg(np.std(data[index][t0_bb:tf_bb]))]
        a_rw_th, b_rw_th = [1/2, lg(np.std(data[index][t0_rw:tf_rw])/np.sqrt(3))]
        # a_bb_th, b_bb_th = [-1/2, lg(np.mean(cov_hh[t0_bb:tf_bb]))]
        # a_rw_th, b_rw_th = [1/2, lg(np.mean(cov_hh[t0_rw:tf_rw])/np.sqrt(3))]
        ax.loglog(T[index], data[index], label="Allan")
        # ax.loglog(T[index], std[index], label="std")
        ax.loglog(T[index][t0_bb:tf_bb], T[index][t0_bb:tf_bb]**a_bb/10**(-b_bb), label=f"bb : {'%.4g'%a_bb}t {'%.4g'%b_bb}")
        ax.loglog(T[index][t0_rw:tf_rw], T[index][t0_rw:tf_rw]**a_rw/10**(-b_rw), label=f"rw : {'%.4g'%a_rw}t {'%.4g'%b_rw}")
        ax.loglog(T[index][t0_bb:tf_bb], T[index][t0_bb:tf_bb]**a_bb_th/10**(-b_bb_th), label=f"bb : {'%.4g'%a_bb_th}t {'%.4g'%b_bb_th}")
        ax.loglog(T[index][t0_rw:tf_rw], T[index][t0_rw:tf_rw]**a_rw_th/10**(-b_rw_th), label=f"rw : {'%.4g'%a_rw_th}t {'%.4g'%b_rw_th}")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(f"Units : {units[j]}")
        ax.legend()

        #Limiter les chiffres significatifs
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
# plt.show()


# #Ploting the Allan deviation
# fig, axs = plt.subplots(1, 2)
# fig.suptitle(f"Déviation d'Allan")
# titles = ["Allan deviation", "Ecart-type"]
# data_list = [data, std]
# labels = ['cap [°]', 'innov_norm_cap [°]']
# for i, ax in enumerate(axs):
#     ax.set_title(titles[i])
#     ax.set_xlabel('Time [s]')
#     for j, l in enumerate(labels):
#         ax.loglog(T[j], data_list[i][j], label=f"{l}")
#     ax.legend()
plt.show()
