import numpy as np
import os

file_path = os.path.dirname(os.path.abspath(__file__))
file_path = file_path+"\gnss_test_station.nma"
trames = np.genfromtxt(file_path, delimiter = ',', dtype = 'str', comments = '$GPZDA', skip_footer = 1)

type_trame = trames[:, 0]
utc = trames[:, 1]
lat = trames[:, 2]
lon = trames[:, 4]
type_pos = np.float64(trames[:, 6])
nb_sat = np.float64(trames[:, 7])
sigma_h = np.float64(trames[:, 8])
alt = np.float64(trames[:, 9])
alt_geoide = np.float64(trames[:, 11])

utc_vis = ['' for i in range(len(utc))]
lat_deg = ['' for i in range(len(lat))]
lat_dms = ['' for i in range(len(lat))]
lon_deg = ['' for i in range(len(lon))]
lon_dms = ['' for i in range(len(lon))]

for i in range(len(utc)):
    utc_vis[i] += utc[i][0:2] + 'h' + utc[i][2:4] + 'm' + utc[i][4:6] + '.' + utc[i][7:] + 's'
    lat_deg[i] += lat[i][0:2] + '.' + str(np.float64(lat[i][2:])/60)[2:8]
    lon_deg[i] += lon[i][0:3] + '.' + str(np.float64(lon[i][3:])/60)[2:8]
    lat_dms[i] += lat[i][0:2] + '°' + lat[i][2:4] + "'" + str(np.float64('0.' + lat[i][5:])*60)[0:4] + '"'
    lon_dms[i] += lon[i][0:3] + '°' + lon[i][3:5] + "'" + str(np.float64('0.' + lon[i][6:])*60)[0:4] + '"' 


#Saving the new data
np.savez("gnss.npz", type_trame=type_trame, utc=utc, lat=lat_deg, lon=lon_deg, type_pos=type_pos, nb_sat=nb_sat, sigma_h=sigma_h, alt=alt, alt_geoide=alt_geoide)

print(len(utc))
print(utc_vis[0])
print(utc_vis[-1])
print(lat[0])
print(lon[0])
print(lat_deg[0])
print(lon_deg[0])
print(lat_dms[0])
print(lon_dms[0])
