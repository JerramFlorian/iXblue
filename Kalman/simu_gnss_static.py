from roblib import *
import os
from tqdm import tqdm
import pyproj as prj
import qrunch



#Importing the data (lat, lon, heading)
file_path = os.path.dirname(os.path.abspath(__file__)) + "\log"
gnss = np.load(file_path + "\gnss_data_qrunch_without_nmea.npz")
allan = np.load(file_path + "\deviation_allan.npz")

lat, lon, cap = gnss["lat_m"], gnss["lon_m"], gnss["heading"]
lat_ref, lon_ref = gnss["lat_ref"], gnss["lon_ref"]
cap = cap - np.mean(cap)
print(lat, lon, cap*180/np.pi)
cov_latlat, cov_lonlon, cov_latlon, cov_hh, cov_pp, cov_hp = gnss["cov_latlat"], gnss["cov_lonlon"], gnss["cov_latlon"], gnss["cov_hh"], gnss["cov_pp"], gnss["cov_hp"]
dev_lat, dev_lon, dev_cap = np.sqrt(3)*10**(-2.931), np.sqrt(3)*10**(-2.568), 0.0005 #sigma bruit


#Defining some useful functions
def f(X, u):
    u1, u2, u3 = u.flatten()
    x1, x2, x3, x4, x5 = X.flatten()
    x_dot = np.array([[x1 + dt*x4],
                      [x2 + dt*x5],
                      [x3 + dt*u1],
                      [x4 + dt*(u2*np.sin(x3) - u3*np.cos(x3))],
                      [x5 + dt*(u2*np.cos(x3) + u3*np.sin(x3))]])
    return x_dot
   
def Kalman(xbar, P, u, y, Q, R, F, G, H):
    # Prédiction
    xbar = f(xbar, u)
    P = F @ P @ F.T + G @ Q @ G.T

    # Correction
    ytilde = y - (H @ xbar)
    S = H @ P @ H.T + R # matrice de covariance d'innovation : covariance des erreurs de mesure entre les mesures réelles et les prédictions de mesure du filtre de Kalman, regarde si on compare des grandeurs du même ordre de grandeur
    innov_norm = sqrtm(np.linalg.inv(S))@ytilde
    K = P @ H.T @ np.linalg.inv(S)
    xbar = xbar + K @ ytilde
    P = P - K @ H @ P

    return xbar, P, ytilde, innov_norm

def draw_ellipse0(ax, c, Γ, a, col,coledge='black'):  # classical ellipse (x-c)T * invΓ * (x-c) <a^2
    # draw_ellipse0(ax,array([[1],[2]]),eye(2),a,[0.5,0.6,0.7])
    A = a * sqrtm(Γ)
    w, v = eig(A)
    v1 = np.array([[v[0, 0]], [v[1, 0]]])
    v2 = np.array([[v[0, 1]], [v[1, 1]]])
    f1 = A @ v1
    f2 = A @ v2
    φ = (arctan2(v1[1, 0], v1[0, 0]))
    α = φ * 180 / np.pi
    e = Ellipse(xy=c, width=2 * norm(f1), height=2 * norm(f2), angle=α)
    ax.add_artist(e)
    e.set_clip_box(ax.bbox)

    e.set_alpha(0.7)
    e.set_facecolor(col)
    e.set_edgecolor(coledge)

    # e.set_fill(False)
    # e.set_alpha(1)
    # e.set_edgecolor(col)
def draw_ellipse_cov(ax,c,Γ,η, col ='blue',coledge='black'): # Gaussian confidence ellipse with artist
    #draw_ellipse_cov(ax,array([[1],[2]]),eye(2),0.9,[0.5,0.6,0.7])
    if (np.linalg.norm(Γ)==0):
        Γ=Γ+0.001*eye(len(Γ[1,:]))
    a=np.sqrt(-2*log(1-η))
    draw_ellipse0(ax, c, Γ, a,col,coledge)

def legende(ax):
    # ax.set_xlim(Xhat[0,0]-8, Xhat[0,0]+8)
    # ax.set_ylim(Xhat[1,0]-8, Xhat[1,0]+8)
    ax.set_title('Filtre de Kalman')
    ax.set_xlabel('x')
    ax.set_ylabel('y')


#Initializing some values
dt = 0.1
display_bot = False
save_data = True
if display_bot:
    ax = init_figure(lon[0]-8, lon[0]+8, lat[0]-8, lat[0]+8)

P = 0.01 * np.eye(5)
Xhat = np.array([[0, 0, 0, 0, 0]]).T
Y = np.array([[0, 0, 0]]).T

sigm_equation = 0.0
Q = np.diag([sigm_equation, sigm_equation, sigm_equation]) #matrice de covariance de bruit pour l'état du système
R = np.diag([dev_lon**2, dev_lat**2, dev_cap**2]) #matrice de covariance de bruit pour les mesure, bruit blanc sinon on estime les bruits non blancs en rajoutant un état dans le Kalman

u = np.array([[0], [0], [0]])

T = []
deno = 1
fr = 1
N = int(np.min([len(lat)//deno, len(lon)//deno, len(cap)//deno]))
# N = 20000
PMatrix = np.zeros((N, P.shape[0]*P.shape[1]))
Innov = np.zeros((N//fr, Y.shape[0]*Y.shape[1]))
Innov_norm = np.zeros((N//fr, Y.shape[0]*Y.shape[1]))
pos = np.zeros((N//fr, 3))

Hk = np.array([[1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0]])

bruit = np.zeros((N//fr, 3))
bruit[:, 0] = np.cumsum(np.random.normal(scale=dev_lon, size=N//fr))
bruit[:, 1] = np.cumsum(np.random.normal(scale=dev_lat, size=N//fr))

# plt.figure()
# plt.plot(bruit[:, 0])
# plt.title("Random walk")

BC = np.zeros((N//fr, 3))
BB = np.array([np.random.normal(scale=dev_cap, size=N//fr) for _ in range(3)]).T
tau = [0.025, 200, 4000]
tau_dt = np.array(tau)
for i in range(1, N//fr):
    BC[i, 0] = BC[i-1, 0]*np.exp(-dt/tau_dt[0]) + np.sqrt((2*tau_dt[0]*dev_cap**2/dt)*(1-np.exp(-2*dt/tau_dt[0])))*BB[i, 0]
    BC[i, 1] = BC[i-1, 1]*np.exp(-dt/tau_dt[1]) + np.sqrt((2*tau_dt[1]*dev_cap**2/dt)*(1-np.exp(-2*dt/tau_dt[1])))*BB[i, 1]
    BC[i, 2] = BC[i-1, 2]*np.exp(-dt/tau_dt[2]) + np.sqrt((2*tau_dt[2]*dev_cap**2/dt)*(1-np.exp(-2*dt/tau_dt[2])))*BB[i, 2]
bc = np.sum(BC, axis=1)
bruit[:, 2] = bc

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
axs[0, 0].plot(BC[:, 0])
axs[0, 0].set_title('Signal de bruit 1')
axs[0, 1].plot(BC[:, 1])
axs[0, 1].set_title('Signal de bruit 2')
axs[1, 0].plot(BC[:, 2])
axs[1, 0].set_title('Signal de bruit 3')
axs[1, 1].plot(bc)
axs[1, 1].set_title('Somme des signaux de bruit')
plt.tight_layout()

# plt.figure()
# plt.plot(bc)
# plt.title("Bruit corrélé")

#Looping our algorithm
for i in tqdm(np.arange(0, N, dt)):
    # Real state

    u1, u2, u3 = u.flatten()
    x1, x2, x3, x4, x5 = Xhat.flatten()

    Fk = np.eye(5) + dt * np.array([[0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 1],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, u2*np.cos(x3) + u3*np.sin(x3), 0, 0],
                                    [0, 0, -u2*np.sin(x3) + u3*np.cos(x3), 0, 0]])

    Gk = dt * np.array([[0, 0, 0],
                        [0, 0, 0],
                        [1, 0, 0],
                        [0, np.sin(x3), -np.cos(x3)],
                        [0, np.cos(x3), np.sin(x3)]])


    if i % fr == 0:
        T.append(i//dt)
        i = int(i//fr)
        Y = np.array([[0, 0, 0]]).T + bruit[int(i//fr), :].reshape(3,1)
        Xhat, P, ytilde, innov_norm = Kalman(Xhat, P, u, Y, Q, R, Fk, Gk, Hk)
        pos[int(i//fr), 0], pos[int(i//fr), 1], pos[int(i//fr), 2] = x1, x2, x3
        for j in range(Y.shape[0]):
            Innov[i, j] = ytilde[j, 0]
            Innov_norm[i, j] = innov_norm[j, 0]
    else:
        Xhat = f(Xhat, u)
        P = Fk @ P @ Fk.T + Gk @ Q @ Gk.T
        

    #Display the results if needed
    if display_bot:
        draw_tank(Xhat)
        # draw_ellipse_cov(ax, Xhat[0:2], 10*P[0:2, 0:2], 0.9, col='black')
        ax.scatter(Xhat[0, 0], Xhat[1, 0], color='red', label = 'Estimation of position', s = 5)
        ax.legend()
        pause(0.001)
        clear(ax)
        legende(ax)
        

    # # Append lists to visualize our covariance model
    # for j in range(P.shape[0]):
    #     for k in range(P.shape[1]):
    #         c = P.shape[0]
    #         PMatrix[i, j+c*k] = P[j, k]
    


if __name__ == "__main__":
    Xhat[2, 0] = Xhat[2, 0]*180/np.pi
    print(Xhat)
    print("Biais sur la 1ère mesure de longitude: {:.3f}".format(pos[-1, 0]))
    print("Biais sur la 1ère mesure de latitude : {:.3f}".format(pos[-1, 1]))
    # T = [int(i/dt+0.5)/3600 for i in T]


    #Ploting some useful contents
    # plt.close()
    # plt.figure()
    # plt.suptitle(f"P(t) Matrix : {N} epoch with step dt={dt}")
    # for i in range(P.shape[0]):
    #     for j in range(P.shape[1]):
    #         ax = plt.subplot2grid((P.shape[0], P.shape[1]), (i, j))
    #         c = P.shape[0]
    #         ax.plot(T, PMatrix[:, i+c*j])
    #         ax.set_xlabel("Time [s]")
    #         ax.set_ylabel("Error [m/rad]")
    #         ax.set_title(f"P_{i},{j}")

    # plt.close("all") 
    T_allan, dev_allan, _ = qrunch.allan_deviation(bruit[:, 1])
    a_th, b_th = np.polyfit(np.log10(T_allan), np.log10(dev_allan), 1) #b=log(sig/sqrt3)
    plt.figure()
    plt.loglog(T_allan, dev_allan, label=f"bruit implémenté : {'%.4g'%a_th}t + {'%.4g'%b_th}")
    plt.loglog(T_allan, dev_lat/np.sqrt(3)*np.sqrt(T_allan), label=f"gabarit : 0.5t + {'%.4g'%np.log10(dev_lat/np.sqrt(3))}")
    plt.loglog(allan["T"], allan["dev_lat"], label="bruit gnss : 0.4453t + -2.931")
    plt.xlabel("Time [s]")
    plt.ylabel("Lat [m]")
    plt.title("Log du bruit")
    plt.legend()

    T_allan, dev_allan, _ = qrunch.allan_deviation(bruit[:, 0])
    a_th, b_th = np.polyfit(np.log10(T_allan), np.log10(dev_allan), 1)
    plt.figure()
    plt.loglog(T_allan, dev_allan, label=f"bruit implémenté : {'%.4g'%a_th}t + {'%.4g'%b_th}")
    plt.loglog(T_allan, dev_lon/np.sqrt(3)*np.sqrt(T_allan), label=f"gabarit : 0.5t + {'%.4g'%np.log10(dev_lon/np.sqrt(3))}")
    plt.loglog(allan["T"], allan["dev_lon"], label="bruit gnss : 0.4001t + -2.568")
    plt.xlabel("Time [s]")
    plt.ylabel("Lon [m]")
    plt.title("Log du bruit")
    plt.legend()

    T_allan, dev_allan, _ = qrunch.allan_deviation(bruit[:, 2])
    plt.figure()
    a_th, b_th = np.polyfit(np.log10(T_allan), np.log10(dev_allan), 1)
    plt.loglog(T_allan, dev_allan, label=f"bruit implémenté : {'%.4g'%a_th}t + {'%.4g'%b_th}")
    cpt = 0
    for to in tau:
        cpt += 1
        exec(f"cap_{cpt} = 2*to*dev_cap**2/T_allan*(1-to/(2*T_allan)*(3-4*np.exp(-T_allan/to)+np.exp(-2*T_allan/to)))")
    plt.loglog(T_allan, np.sqrt(sum(eval(f'cap_{cpt}') for cpt in range(1, len(tau)+1))), label=f"gabarit : {tau}")
    plt.loglog(allan["T"], allan["dev_cap"], label="bruit gnss : à définir")
    plt.xlabel("Time [s]")
    plt.ylabel("Cap [rad]")
    plt.title("Log du bruit")
    plt.legend()

    # plt.close("all")
    plt.figure()
    titles = ["Lon [m]", "Lat [m]", "Cap [rad]"]
    plt.suptitle(f"Innovation Matrix : {N} epoch with step dt={dt}")
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            ax = plt.subplot2grid((Y.shape[0], 2*Y.shape[1]), (i, j))
            c = Y.shape[0]
            ax.plot(T, Innov[:, i+c*j])
            ax.set_xlabel("Time [s]")
            ax.set_ylabel(f"{titles[i]}")
            if j==0:
                ax = plt.subplot2grid((Y.shape[0], 2*Y.shape[1]), (i, 1))
            else:
                 ax = plt.subplot2grid((Y.shape[0], 2*Y.shape[1]), (i, 2*j))               
            c = Y.shape[0]
            ax.plot(T, Innov_norm[:, i+c*j])
            ax.set_xlabel("Time [s]")
            ax.set_ylabel(f"{titles[i]}")
    # plt.show()

    fig, axs = plt.subplots(3, 1, figsize=(8, 8))  # Créer une figure avec 2 subplots verticaux
    fig.suptitle("Evolution de la position avec le bruit")
    axs[0].plot(pos[:, 0], label="longitude")
    axs[0].plot(bruit[:, 0], label="bruit longitude")
    axs[0].set_xlabel("Time [s]")
    axs[0].set_ylabel("Lon [m]")
    axs[0].legend()
    axs[1].plot(pos[:, 1], label="latitude")
    axs[1].plot(bruit[:, 1], label="bruit latitude")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Lat [m]")
    axs[1].legend()
    axs[2].plot(pos[:, 2], label="latitude")
    axs[2].plot(bruit[:, 2], label="bruit latitude")
    axs[2].set_xlabel("Time [s]")
    axs[2].set_ylabel("Cap [rad]")
    axs[2].legend()
    plt.tight_layout()  # Ajuster les sous-graphiques pour les rendre plus lisibles
    plt.show()


    if save_data == True:
        np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)) + "\log", "innovation_normalisee.npz"), innovation=Innov, innovation_normalisee=Innov_norm)
