import pyproj as prj
import os
import numpy as np


#Importing the NMEA data
file_path = os.path.dirname(os.path.abspath(__file__)) + "\..\log"
gnss = np.load(file_path + "\gnss_data.npz")
lat_deg, lon_deg = gnss["lat"], gnss["lon"]


#Projecting the NMEA data
rep_base = prj.CRS("epsg:4919")
proj = prj.CRS("epsg:4326") #Lambert93 : 2154
t = prj.Transformer.from_crs(rep_base, proj, always_xy=True)
print(rep_base.datum, proj.datum)
E, N = t.transform(lon_deg, lat_deg)
# print(E, N)


#Notes
"""
coordonnées vers l'est en premier, vers le nord en
second (c'est-à-dire (E, N) en projection, mais surtout longitude, latitude - dans
cet ordre - en coordonnées géographiques).
mon GNSS double antenne m'affiche un datum combiné WSG84/ITRS, comment je fais pour projeter cela en coordonnées cartésiennes Lambert93 avec le module CRS de pyproj ?
"""