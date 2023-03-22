import numpy as np
import os


#Importing the data
file_path = os.path.dirname(os.path.abspath(__file__))
trames_path = file_path + "\double_antenne.nma"
att_path = file_path + "\double_antenne.sbf_SBF_AttEuler1.txt"

trames = np.genfromtxt(trames_path, delimiter = ',', dtype = 'str', skip_footer = 1)
att = np.genfromtxt(att_path, delimiter = ',', dtype = 'str', skip_header = 10, skip_footer = 1)


#Extracting NMEA data
type_trame = trames[:, 0]
utc = trames[:, 1]
lat = trames[:, 2] #N
lon = trames[:, 4] #E
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
    lat_deg[i] = np.float64(lat_deg[i])
    lon_deg[i] = np.float64(lon_deg[i])


#Extracting attitude data
heading = np.float64(att[:, 17])*np.pi/180
pitch = np.float64(att[:, 18])


#Saving the new data
np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gnss_data.npz"), type_trame=type_trame, utc=utc, lat=lat_deg, lon=lon_deg, heading = heading, pitch = pitch, type_pos=type_pos, nb_sat=nb_sat, sigma_h=sigma_h, alt=alt, alt_geoide=alt_geoide, dtype=float)


#Printing some results
print(len(utc))
print(len(heading))
print(utc_vis[0])
print(utc_vis[-1])
print(lat[0])
print(lon[0])
print(lat_deg[0])
print(lon_deg[0])
print(lat_dms[0])
print(lon_dms[0])
print(heading[0])
print(pitch[0])