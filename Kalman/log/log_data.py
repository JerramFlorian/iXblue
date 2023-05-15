import numpy as np
import os
import qrunch
import pyproj as prj


#Importing the data
file_path = os.path.dirname(os.path.abspath(__file__)) + "\sbf\moving\SBAS"
trames_path = file_path + "\long_acquisition_rov.nma"
pos_path = file_path + "\long_acquisition_rov.sbf_SBF_PVTGeodetic2.txt"
att_path = file_path + "\long_acquisition_rov.sbf_SBF_AttEuler1.txt"
cov_att_path = file_path + "\long_acquisition_rov.sbf_SBF_AttCovEuler1.txt"
cov_pos_path = file_path + "\long_acquisition_rov.sbf_SBF_PosCovGeodetic1.txt"
aux_path = file_path + "\long_acquisition_rov.sbf_SBF_AuxAntPositions1.txt"

trames = np.genfromtxt(trames_path, delimiter = ',', dtype = 'str', skip_footer = 3)
pos = np.genfromtxt(pos_path, delimiter = ',', dtype = 'str', skip_footer = 3)
att = np.genfromtxt(att_path, delimiter = ',', dtype = 'str', skip_footer = 3)
cov_att = np.genfromtxt(cov_att_path, delimiter = ',', dtype = 'str', skip_footer = 3)
cov_pos = np.genfromtxt(cov_pos_path, delimiter = ',', dtype = 'str', skip_footer = 3)
aux = np.genfromtxt(aux_path, delimiter = ',', dtype = 'str', skip_footer = 3)


#Extracting NMEA data
def NMEA_data():
    type_trame = trames[:, 0]
    utc = trames[:, 1]
    lat = trames[:, 2] #N
    lon = trames[:, 4] #E
    type_pos = np.float64(trames[:, 6])
    nb_sat = np.float64(trames[:, 7])
    sigma_h = np.float64(trames[:, 8])
    alt = np.float64(trames[:, 9])
    alt_geoide = np.float64(trames[:, 11])

    utc_vis = ['' for _ in range(len(utc))]
    lat_deg = ['' for _ in range(len(lat))]
    lat_dms = ['' for _ in range(len(lat))]
    lon_deg = ['' for _ in range(len(lon))]
    lon_dms = ['' for _ in range(len(lon))]

    for i in range(len(utc)):
        utc_vis[i] += utc[i][0:2] + 'h' + utc[i][2:4] + 'm' + utc[i][4:6] + '.' + utc[i][7:] + 's'
        lat_deg[i] += lat[i][0:2] + '.' + str(np.float64(lat[i][2:])/60)[2:8]
        lon_deg[i] += lon[i][0:3] + '.' + str(np.float64(lon[i][3:])/60)[2:8]
        lat_dms[i] += lat[i][0:2] + '°' + lat[i][2:4] + "'" + str(np.float64('0.' + lat[i][5:])*60)[0:4] + '"'
        lon_dms[i] += lon[i][0:3] + '°' + lon[i][3:5] + "'" + str(np.float64('0.' + lon[i][6:])*60)[0:4] + '"' 
        lat_deg[i] = np.float64(lat_deg[i])
        lon_deg[i] = np.float64(lon_deg[i])

    return(utc, lat, lon, lat_deg, lon_deg, nb_sat, sigma_h, alt, alt_geoide)
def NMEA_qrunch():
    return(qrunch.load_gnssnmea(trames_path))

#Extracting position data
def position_data(): #radian
    try:
        lat = np.float64(pos[:, 15])
        lon = np.float64(pos[:, 16])
        alt = np.float64(pos[:, 17])
    except:
        print("Warning : position data weren't rightly saved !")
        cpt = 0
        lat = [] ; lon = [] ; alt = []
        for i in range(len(pos[:, 15])):
            if pos[i, 15] != '' or pos[i, 16] != '' or pos[i, 17] != '':
                lat.append(float(pos[i, 15]))
                lon.append(float(pos[i, 16]))
                alt.append(float(pos[i, 17]))
            else:
                cpt += 1
        print("Number of erroneous data : ", cpt)
    return(lat, lon, alt)

#Extracting attitude data
def attitude_data(): #radian
    try:
        heading = np.float64(att[:, 17])*np.pi/180
        pitch = np.float64(att[:, 18])*np.pi/180
    except:
        print("Warning : attitude data weren't rightly saved !")
        cpt = 0
        heading = [] ; pitch = []
        for i in range(len(att[:, 17])):
            if att[i, 17] != '' or att[i, 18] != '':
                heading.append(float(att[i, 17])*np.pi/180)
                pitch.append(float(att[i, 18])*np.pi/180)
            else:
                cpt += 1
        print("Number of erroneous data : ", cpt)
    return(heading, pitch)

#Extracting covariance position data
def covariance_pos(): #m²
    try:
        cov_latlat = np.float64(cov_pos[:, 15])
        cov_lonlon = np.float64(cov_pos[:, 16])
        cov_latlon = np.float64(cov_pos[:, 19])
    except:
        print("Warning : position covariance data weren't rightly saved !")
        cpt = 0
        cov_latlat = [] ; cov_lonlon = [] ; cov_latlon = []
        for i in range(len(cov_pos[:, 15])):
            if cov_pos[i, 15] != '' or cov_pos[i, 16] != '' or cov_pos[i, 19] != '':
                cov_latlat.append(float(cov_pos[i, 15]))
                cov_lonlon.append(float(cov_pos[i, 16]))
                cov_latlon.append(float(cov_pos[i, 19]))
            else:
                cpt += 1
        print("Number of erroneous data : ", cpt)
    return(cov_latlat, cov_lonlon, cov_latlon)

#Extracting covariance attitude data
def covariance_att(): #rad²
    try:
        cov_hh = np.float64(cov_att[:, 15])*(np.pi/180)**2
        cov_pp = np.float64(cov_att[:, 16])*(np.pi/180)**2
        cov_hp = np.float64(cov_att[:, 18])*(np.pi/180)**2
    except:
        print("Warning : attitude covariance data weren't rightly saved !")
        cpt = 0
        cov_hh = [] ; cov_pp = [] ; cov_hp = []
        for i in range(len(cov_att[:, 15])):
            if cov_att[i, 15] != '' or cov_att[i, 16] != '' or cov_att[i, 18] != '':
                cov_hh.append(float(cov_att[i, 15])*(np.pi/180)**2)
                cov_pp.append(float(cov_att[i, 16])*(np.pi/180)**2)
                cov_hp.append(float(cov_att[i, 18])*(np.pi/180)**2)
            else:
                cpt += 1
        print("Number of erroneous data : ", cpt)
    return(cov_hh, cov_pp, cov_hp)

#Extracting Delta data
def Delta(): #m
    try:
        DE = np.float64(aux[:, -6])
        DN = np.float64(aux[:, -5])
        DU = np.float64(aux[:, -4])
    except:
        print("Warning : delta data weren't rightly saved !")
        cpt = 0
        DE = [] ; DN = [] ; DU = []
        for i in range(len(aux[:, -6])):
            if aux[i, -6] != '' or aux[i, -5] != '' or aux[i, -4] != '':
                DE.append(float(aux[i, -6]))
                DN.append(float(aux[i, -5]))
                DU.append(float(aux[i, -4]))
            else:
                cpt += 1
        print("Number of erroneous data : ", cpt)
    return(DE, DN, DU)

#Calculating the cap
def calc_att(be, bn, bu): #deg
    be, bn, bu = np.array(be), np.array(bn), np.array(bu)
    head = np.arccos(bn/np.sqrt(bn**2+be**2))*180/np.pi
    pitch = np.arctan(bu/np.sqrt(bn**2+be**2))*180/np.pi
    return(360-head, pitch)

#Projecting the NMEA data
def proj2(lon_rad, lat_rad): #radian --> m
    rep_geo = prj.CRS("EPSG:4326")
    proj = prj.CRS("EPSG:2154")
    t = prj.Transformer.from_crs(rep_geo, proj, always_xy=True)
    print(rep_geo.datum, proj.datum)
    E, N = t.transform(lon_rad, lat_rad)
    return(np.array(E), np.array(N))

#Projecting the NMEA data
def proj(lon_rad, lat_rad): #radian --> m
    lambert93 = prj.Proj("+init=EPSG:2154")
    x, y = prj.transform(prj.Proj("+proj=longlat +datum=WGS84"), lambert93, lon_rad, lat_rad)
    return(x, y)


if __name__ == "__main__":
    #Manual method
    # print("----- Extracting the manual data -----")
    # utc, lat, lon, lat_deg, lon_deg, nb_sat, sigma_h, alt, alt_geoide = NMEA_data()
    # heading, pitch = attitude_data()
    # cov_latlat, cov_lonlon, cov_latlon = covariance_pos()
    # cov_hh, cov_pp, cov_hp = covariance_att()
    # np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gnss_data_manual.npz"), utc=utc, lat=lat_deg, lon=lon_deg, heading=heading, pitch=pitch, cov_latlat=cov_latlat, cov_lonlon=cov_lonlon, cov_latlon=cov_latlon, cov_hh=cov_hh, cov_pp=cov_pp, cov_hp=cov_hp, nb_sat=nb_sat, sigma_h=sigma_h, alt=alt, alt_geoide=alt_geoide, dtype=float)
    # print("----- Saving the manual data -----")

    # #Qrunch method with NMEA
    # print("----- Extracting the qrunch (with NMEA) data -----")
    # time, lat, lon, alt, easting, northing, zone_number, zone_letter, gps_quality, nb_sat, h_dilution = NMEA_qrunch()
    # heading, pitch = attitude_data()
    # cov_latlat, cov_lonlon, cov_latlon = covariance_pos()
    # cov_hh, cov_pp, cov_hp = covariance_att()
    # np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gnss_data_qrunch_with_nmea.npz"), time=time, lat=lat, lon=lon, alt=alt, easting=easting, northing=northing, heading=heading, pitch=pitch, cov_latlat=cov_latlat, cov_lonlon=cov_lonlon, cov_latlon=cov_latlon, cov_hh=cov_hh, cov_pp=cov_pp, cov_hp=cov_hp, zone_number=zone_number, zone_letter=zone_letter, gps_quality=gps_quality, nb_sat=nb_sat, h_dilution=h_dilution)
    # print("----- Saving the qrunch (with NMEA) data -----")

    #Qrunch method without NMEA
    print("\n----- Extracting the qrunch (without NMEA) data -----")
    lat, lon, alt = position_data()
    heading, pitch = attitude_data()
    cov_latlat, cov_lonlon, cov_latlon = covariance_pos()
    cov_hh, cov_pp, cov_hp = covariance_att()
    DE, DN, DU = Delta()
    head, pit = calc_att(DE, DN, DU)
    lon_m, lat_m = proj2(lon, lat)
    lon_ref, lat_ref = proj2(0.036023633030, 0.853455632381)
    lon_m, lat_m = lon_m - lon_ref, lat_m - lat_ref
    print(np.shape(lon_m))
    lat_m=lat_m[:18000] ; lon_m=lon_m[:18000] ; heading=heading[:18000] ; DE=DE[:18000] ; DN=DN[:18000] ; DU=DU[:18000]
    print(np.shape(lon_m))
    np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gnss_data_qrunch_without_nmea.npz"), lat_ref=lat_ref, lon_ref=lon_ref, alt=alt, lat_m=lat_m, lon_m=lon_m, heading=heading, pitch=pitch, calc_head=head, calc_pitch=pit, cov_latlat=cov_latlat, cov_lonlon=cov_lonlon, cov_latlon=cov_latlon, cov_hh=cov_hh, cov_pp=cov_pp, cov_hp=cov_hp, DE=DE, DN=DN, DU=DU)
    print("----- Saving the qrunch (without NMEA) data -----\n")
