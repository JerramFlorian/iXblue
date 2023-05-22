from roblib import *
import os
from tqdm import tqdm
import qrunch



#Importing the data (lat, lon, heading)
file_path = os.path.dirname(os.path.abspath(__file__)) + "\log"
gnss = np.load(file_path + "\gnss_data_qrunch_without_nmea.npz")
allan = np.load(file_path + "\deviation_allan.npz")
# print(allan["T"])
T_ref = allan["T"][:np.where(allan["T"]==2877)[0][0]]/60
dev_lon_ref, dev_lat_ref, dev_cap_ref = allan["dev_lon"][:len(T_ref)], allan["dev_lat"][:len(T_ref)], allan["dev_cap"][:len(T_ref)]

lat, lon, cap = gnss["lat_m"], gnss["lon_m"], gnss["heading"] #ici lat et lon sont absolues (pos = pos - pos_ref)
lat_ref, lon_ref = gnss["lat_ref"], gnss["lon_ref"]
N = 8000
cap = cap - np.mean(cap)
lat=lat[:N] ; lon=lon[:N] ; cap=cap[:N]
print(lat, lon, cap*180/np.pi)
dev_lat, dev_lon, dev_cap = 2*10**-3, 4.5*10**-3, 5*10**-4 #écart-type bruit


#Defining some useful functions
def f(X, u, bruit_biais):
    u1, u2, u3, u4, u5, u6 = u.flatten()
    x1, x2, x3, x4, x5, x6, x7, x8 = X.flatten()
    b1, b2, b3 = bruit_biais.flatten()
    x_dot = np.array([[x1 + dt*x4],
                      [x2 + dt*x5],
                      [x3 + dt*u1],
                      [x4 + dt*(u2*np.sin(x3) - u3*np.cos(x3))],
                      [x5 + dt*(u2*np.cos(x3) + u3*np.sin(x3))],
                      [x6 + dt*b1],
                      [x7 + dt*b2],
                      [b3]])
    return x_dot

def Kalman(xbar, P, u, y, Q, R, F, G, H, bruit_biais):
    # Prédiction
    xbar = f(xbar, u, bruit_biais)
    P = F @ P @ F.T + G @ Q @ G.T

    # Correction
    ytilde = y - (H @ xbar)
    S = H @ P @ H.T + R # matrice de covariance d'innovation : covariance des erreurs de mesure entre les mesures réelles et les prédictions de mesure du filtre de Kalman, regarde si on compare des grandeurs du même ordre de grandeur
    innov_norm = sqrtm(np.linalg.inv(S))@ytilde
    K = P @ H.T @ np.linalg.inv(S)
    xbar = xbar + K @ ytilde
    P = P - K @ H @ P

    return xbar, P, ytilde, innov_norm

def se1():
    return(0*np.exp(-dt/tau_dt[0]) + np.exp(-dt/tau_dt[1]) + 0*np.exp(-dt/tau_dt[2]))

def se2():
    return(0*np.sqrt(1-np.exp(-2*dt/tau_dt[0])) + np.sqrt(1-np.exp(-2*dt/tau_dt[1])) + 0*np.sqrt(1-np.exp(-2*dt/tau_dt[2])))

def RMS(values):
    squared_sum = np.sum(np.fromiter((x**2 for x in values), np.float64))
    mean_squared = squared_sum/len(values)
    rms = np.sqrt(mean_squared)
    return rms


#Initializing some values
dt = 0.1
save_data = True

p0, p1 = 10**-6, 10**-2
P = np.diag([p0**2, p0**2, p0**2, p0**2, p0**2, p1**2, p1**2, p1**2])
Xhat = np.array([[0, 0, 0, 0, 0, 0, 0, 0]]).T
Y = np.array([[0, 0, 0]]).T

q0 = 10**-8
Q = np.diag([q0**2, q0**2, q0**2, dev_lon**2, dev_lat**2, dev_cap**2]) #matrice de covariance de bruit pour l'état du système
R = np.diag([dev_lon**2, dev_lat**2, dev_cap**2]) #matrice de covariance de bruit pour les mesure, bruit blanc sinon on estime les bruits non blancs en rajoutant un état dans le Kalman

u = np.array([[0], [0], [0], [0], [0], [0]])

T = []
PMatrix = np.zeros((N, P.shape[0]*P.shape[1]))
Innov = np.zeros((N, Y.shape[0]*Y.shape[1]))
Innov_norm = np.zeros((N, Y.shape[0]*Y.shape[1]))
obs = np.zeros((N, Y.shape[0]*Y.shape[1]))
pos = np.zeros((N, 3))
biais = np.zeros((N, 3))

Hk = np.array([[1, 0, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 0, 1]])

bruit = np.zeros((N, 3))
bruit[:, 0] = np.cumsum(np.random.normal(scale=dev_lon, size=N))
bruit[:, 1] = np.cumsum(np.random.normal(scale=dev_lat, size=N))

BC = np.zeros((N, 3))
BB = np.array([np.random.normal(scale=np.sqrt(dev_cap), size=N) for _ in range(3)]).T
tau = [200, 4000]
tau_dt = np.array(tau)#*dt
for i in range(5, N):
    BC[i, 0] = BC[i-1, 0]*np.exp(-dt/tau_dt[0]) + np.sqrt(dev_cap)*np.sqrt(1-np.exp(-2*dt/tau_dt[0]))*BB[i, 0]
    BC[i, 1] = BC[i-1, 1]*np.exp(-dt/tau_dt[1]) + np.sqrt(dev_cap)*np.sqrt(1-np.exp(-2*dt/tau_dt[1]))*BB[i, 1]
    BC[i, 2] = BC[i-1, 2]*np.exp(-dt/tau_dt[2]) + np.sqrt(dev_cap)*np.sqrt(1-np.exp(-2*dt/tau_dt[2]))*BB[i, 2]
bc = np.sum(BC, axis=1)
bruit[:, 2] = bc

bruit_blanc_biais = np.array([np.random.normal(scale=dev_lon, size=N), np.random.normal(scale=dev_lat, size=N)]).T
BB = np.array([np.random.normal(scale=np.sqrt(dev_cap), size=N) for _ in range(3)]).T
for i in range(1, N):
    BC[i, 0] = BC[i-1, 0]*np.exp(-dt/tau_dt[0]) + np.sqrt(dev_cap)*np.sqrt(1-np.exp(-2*dt/tau_dt[0]))*BB[i, 0]
    BC[i, 1] = BC[i-1, 1]*np.exp(-dt/tau_dt[1]) + np.sqrt(dev_cap)*np.sqrt(1-np.exp(-2*dt/tau_dt[1]))*BB[i, 1]
    BC[i, 2] = BC[i-1, 2]*np.exp(-dt/tau_dt[2]) + np.sqrt(dev_cap)*np.sqrt(1-np.exp(-2*dt/tau_dt[2]))*BB[i, 2]
bruit_correle_biais = np.sum(BC, axis=1)


#Looping our algorithm
for i in tqdm(np.arange(0, N, dt)):
    # Real state
    u1, u2, u3, u4, u5, u6 = u.flatten()
    x1, x2, x3, x4, x5, x6, x7, x8 = Xhat.flatten()

    Fk = np.eye(8) + dt * np.array([[0, 0, 0, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, u2*np.cos(x3) + u3*np.sin(x3), 0, 0, 0, 0, 0],
                                    [0, 0, -u2*np.sin(x3) + u3*np.cos(x3), 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, (se1()-1)/dt]]) 

    Gk = dt * np.array([[0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0],
                        [0, np.sin(x3), -np.cos(x3), 0, 0, 0],
                        [0, np.cos(x3), np.sin(x3), 0, 0, 0],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, np.sqrt(dev_cap)*se2()/dt]])

    if i % 1 == 0: # fréquence=1
        i = int(i)
        T.append(i)
        Y = np.array([[0, 0, 0]]).T + bruit[i, :].reshape(3,1)
        pos[i, 0], pos[i, 1], pos[i, 2] = x1, x2, x3
        biais[i, 0], biais[i, 1], biais[i, 2] = x6, x7, x8
        bruit_biais = np.array([[bruit_blanc_biais[i, 0], bruit_blanc_biais[i, 1], bruit_correle_biais[i]]]).T
        Xhat, P, ytilde, innov_norm = Kalman(Xhat, P, u, Y, Q, R, Fk, Gk, Hk, bruit_biais)
        for j in range(Y.shape[0]):
            Innov[i, j] = ytilde[j, 0]
            Innov_norm[i, j] = innov_norm[j, 0]
            obs[i, j] = Y[j, 0]
    else: #propagation
        Xhat = f(Xhat, u, np.array([0, 0, 0]).T)
        P = Fk @ P @ Fk.T + Gk @ Q @ Gk.T
    

    # # Append lists to visualize our covariance model
    i = int(i)
    for j in range(P.shape[0]):
        for k in range(P.shape[1]):
            c = P.shape[0]
            PMatrix[i, j+c*k] = P[j, k]
    

if __name__ == "__main__":
    Xhat[2, 0] = Xhat[2, 0]*180/np.pi
    print('\nXhat : ', Xhat) #en degré donc
    print("\nErreur absolue minimale sur la longitutde : {:.3f}".format(np.min(np.abs(lon))))
    print("Erreur absolue minimale sur la latitude : {:.3f}".format(np.min(np.abs(lat))))
    print("\nErreur absolue maximale sur la longitutde: {:.3f}".format(np.max(np.abs(lon))))
    print("Erreur absolue maximale sur la latitude : {:.3f}".format(np.max(np.abs(lat))))
    print("\nRMS de l'erreur absolue sur la longitude : {:.3f}".format(RMS(lon))) # RMS fort ==> grande fluctuation 
    print("RMS de l'erreur absolue sur la latitude : {:.3f}".format(RMS(lat)))
    print("\nBiais sur la 1ère mesure de longitude: {:.3f}".format(np.abs(lon[0])))
    print("Biais sur la 1ère mesure de latitude : {:.3f}".format(np.abs(lat[1])))
    print("\nPrécision moyenne sur la longitude : {:.3f}".format(np.mean(lon)))
    print("Précision moyenne sur la latitude : {:.3f}".format(np.mean(lat)))
    print("\nEcart-type moyen sur la longitude : {:.3f}".format(np.std(lon))) # distribution gaussienne donc RMS=STD ici
    print("Ecart-type moyen sur la latitude : {:.3f}\n".format(np.std(lat)))
    T = [i/3600 for i in T]


    #Ploting some useful contents
    # plt.figure()
    # plt.suptitle(f"P(t) Matrix : {N} epoch with step dt={dt}")
    # for i in range(P.shape[0]):
    #     for j in range(P.shape[1]):
    #         ax = plt.subplot2grid((P.shape[0], P.shape[1]), (i, j))
    #         c = P.shape[0]
    #         ax.plot(T, PMatrix[:, i+c*j])
    #         ax.set_xlabel("Time [h]")
    #         ax.set_ylabel("Error [m/rad]")
    #         ax.set_title(f"P_{i},{j}")
 
    plt.figure()
    plt.suptitle(f"P(t) Matrix : {N} epoch with step dt={dt}")
    for i in range(2):
        for j in range(P.shape[1]//2):
            ax = plt.subplot2grid((2, P.shape[1]//2), (i, j))
            c = P.shape[1]//2
            ax.plot(T, PMatrix[:, i*c+j])
            ax.set_xlabel("Time [h]")
            ax.set_ylabel("Error [m/rad]")
            ax.set_title(f"P_{i*c+j},{i*c+j}")

    T_allan, dev_allan, _ = qrunch.allan_deviation(bruit[:, 1])
    a_th, b_th = np.polyfit(np.log10(T_allan), np.log10(dev_allan), 1) #b=log(sig/sqrt3)
    plt.figure()
    plt.loglog(T_allan/60, dev_allan, label=f"bruit implémenté : {'%.4g'%a_th}t + {'%.4g'%b_th}")
    plt.loglog(T_allan/60, dev_lat/np.sqrt(3)*np.sqrt(T_allan), label=f"gabarit : 0.5t + {'%.4g'%np.log10(dev_lat/np.sqrt(3))}")
    plt.loglog(T_ref, dev_lat_ref, label="bruit gnss : 0.4453t + -2.931")
    plt.xlabel("Time [min]")
    plt.ylabel("Lat [m]")
    plt.title("Log du bruit")
    plt.legend()

    T_allan, dev_allan, _ = qrunch.allan_deviation(bruit[:, 0])
    a_th, b_th = np.polyfit(np.log10(T_allan), np.log10(dev_allan), 1)
    plt.figure()
    plt.loglog(T_allan/60, dev_allan, label=f"bruit implémenté : {'%.4g'%a_th}t + {'%.4g'%b_th}")
    plt.loglog(T_allan/60, dev_lon/np.sqrt(3)*np.sqrt(T_allan), label=f"gabarit : 0.5t + {'%.4g'%np.log10(dev_lon/np.sqrt(3))}")
    plt.loglog(T_ref, dev_lon_ref, label="bruit gnss : 0.4001t + -2.568")
    plt.xlabel("Time [min]")
    plt.ylabel("Lon [m]")
    plt.title("Log du bruit")
    plt.legend()

    T_allan, dev_allan, _ = qrunch.allan_deviation(bruit[:, 2])
    plt.figure()
    plt.loglog(T_allan/60, dev_allan, label="bruit implémenté")
    cpt = 0
    for to in tau:
        cpt += 1
        exec(f"cap_{cpt} = 2*to*dev_cap**2/T_allan*(1-to/(2*T_allan)*(3-4*np.exp(-T_allan/to)+np.exp(-2*T_allan/to)))")
    plt.loglog(T_allan/60, np.sqrt(np.sum(eval(f'cap_{cpt}') for cpt in range(1, len(tau)+1))), label=f"gabarit : {tau}")
    plt.loglog(T_ref, dev_cap_ref, label="bruit gnss")
    plt.xlabel("Time [min]")
    plt.ylabel("Cap [rad]")
    plt.title("Log du bruit")
    plt.legend()

    plt.figure()
    titles = ["Lon [m]", "Lat [m]", "Cap [rad]"]
    plt.suptitle(f"Innovation Matrix : {N} epoch with step dt={dt}")
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            ax = plt.subplot2grid((Y.shape[0], 2*Y.shape[1]), (i, j))
            c = Y.shape[0]
            ax.plot(T, Innov[:, i+c*j])
            ax.set_xlabel("Time [h]")
            ax.set_ylabel(f"{titles[i]}")
            if j==0:
                ax = plt.subplot2grid((Y.shape[0], 2*Y.shape[1]), (i, 1))
            else:
                 ax = plt.subplot2grid((Y.shape[0], 2*Y.shape[1]), (i, 2*j))               
            c = Y.shape[0]
            ax.plot(T, Innov_norm[:, i+c*j])
            ax.set_xlabel("Time [h]")
            ax.set_ylabel(f"{titles[i]}")

    fig, axs = plt.subplots(3, 1, figsize=(8, 8))
    fig.suptitle("Evolution de la position avec le bruit")
    axs[0].plot(T, pos[:, 0], label="sortie kalman longitude")
    axs[0].plot(T, obs[:, 0], label="simulation gnss data observation longitude")
    # axs[0].plot(T, lon, label="vraie longitude")
    axs[0].plot(T, biais[:, 0], label="biais estimé longitude", color='purple')
    axs[0].set_xlabel("Time [h]")
    axs[0].set_ylabel("Lon [m]")
    axs[0].legend()
    axs[1].plot(T, pos[:, 1], label="sortie kalman latitude")
    axs[1].plot(T, obs[:, 1], label="simulation gnss data observation latitude")
    # axs[1].plot(T, lat, label="vraie latitude")
    axs[1].plot(T, biais[:, 1], label="biais estimé latitude", color='purple')
    axs[1].set_xlabel("Time [h]")
    axs[1].set_ylabel("Lat [m]")
    axs[1].legend()
    axs[2].plot(T, pos[:, 2], label="sortie kalman cap")
    axs[2].plot(T, obs[:, 2], label="simulation gnss data observationcap")
    # axs[2].plot(T, cap, label="vrai cap")
    axs[2].plot(T, biais[:, 2], label="biais estimé cap", color='purple')
    axs[2].set_xlabel("Time [h]")
    axs[2].set_ylabel("Cap [rad]")
    axs[2].legend()
    plt.tight_layout() # Ajuster les sous-graphiques pour les rendre plus lisibles
    plt.show()


    if save_data == True:
        np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)) + "\log", "innovation_normalisee.npz"), innovation=Innov, innovation_normalisee=Innov_norm)
