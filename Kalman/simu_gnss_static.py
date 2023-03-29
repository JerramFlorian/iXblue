from roblib import *
import os
from tqdm import tqdm
import pyproj as prj


#Importing the data (lat, lon, heading)
file_path = os.path.dirname(os.path.abspath(__file__)) + "\log"
gnss = np.load(file_path + "\gnss_data_qrunch_without_nmea.npz")

lat_deg, lon_deg, cap = gnss["lat"], gnss["lon"], gnss["heading"]
cov_latlat, cov_lonlon, cov_latlon, cov_hh, cov_pp, cov_hp = gnss["cov_latlat"], gnss["cov_lonlon"], gnss["cov_latlon"], gnss["cov_hh"], gnss["cov_pp"], gnss["cov_hp"]


#Projecting the NMEA data
rep_geo = prj.CRS("EPSG:4326")
proj = prj.CRS("EPSG:2154")
t = prj.Transformer.from_crs(rep_geo, proj, always_xy=True)
print(rep_geo.datum, proj.datum)
E, N = t.transform(lon_deg, lat_deg)


#Defining some useful functions
def f(X, u):
    u1, u2, u3 = u.flatten()
    x1, x2, x3, x4, x5 = X.flatten()
    x_dot = np.array([[x1 + dt*x4],
                      [x2 + dt*x5],
                      [x3 + dt*u1],
                      [x4 + dt*(u2*np.cos(x3) + u3*np.sin(x3))],
                      [x5 + dt*(u2*np.sin(x3) - u3*np.cos(x3))]])
    return x_dot

def Kalman(xbar, P, u, y, Q, R, F, G, H):
    xbar = f(xbar, u) #+ mvnrnd1(Gk @ Q @ Gk.T) #+ bruit(xbar, 0, 0.1)
    P = F @ P @ F.T + G @ Q @ G.T

    # Correction
    ytilde = y - (H @ xbar)
    # print("yt : ", ytilde)
    S = H @ P @ H.T + R
    innov_norm = sqrtm(np.linalg.inv(S))@ytilde
    K = P @ H.T @ np.linalg.inv(S)
    xbar = xbar + K @ ytilde
    P = P - K @ H @ P

    return xbar, P, ytilde, innov_norm

def bruit(M, mean, std):
    B = np.random.normal(mean, std, size=(M.shape[0], M.shape[1]))
    # print('bruit : ', B)
    return(B)

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
save_data = False
if display_bot:
    ax = init_figure(lon_deg[0]-8, lon_deg[0]+8, lat_deg[0]-8, lat_deg[0]+8)

P = 0.01 * np.eye(5)
Xhat = np.array([[lon_deg[0], lat_deg[0], cap[0], 0, 0]]).T
Y = np.array([[lon_deg[0], lat_deg[0], cap[0]]]).T

sigm_equation = 0.001
sigm_measure = 0.001
Q = np.diag([sigm_equation, sigm_equation, sigm_equation])
R = np.diag([sigm_measure, sigm_measure, sigm_measure])

u = np.array([[0], [0], [0]])

T = []
N = np.min([len(lat_deg), len(lon_deg), len(cap)])
PMatrix = np.zeros((N, P.shape[0]*P.shape[1]))
Innov = np.zeros((N, Y.shape[0]*Y.shape[1]))
Innov_norm = np.zeros((N, Y.shape[0]*Y.shape[1]))


#Looping our algorithm
for i in tqdm(np.arange(0, N*dt, dt)):
    # Real state

    u1, u2, u3 = u.flatten()
    x1, x2, x3, x4, x5 = Xhat.flatten()

    Hk = np.array([[1, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0],
                   [0, 0, 1, 0, 0]])

    Fk = np.eye(5) + dt * np.array([[0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 1],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, -u2*np.sin(x3) + u3*np.cos(x3), 0, 0],
                                    [0, 0, u2*np.cos(x3) + u3*np.sin(x3), 0, 0]])

    Gk = dt * np.array([[0, 0, 0],
                        [0, 0, 0],
                        [1, 0, 0],
                        [0, np.cos(x3), np.sin(x3)],
                        [0, np.sin(x3), -np.cos(x3)]])


    Y = np.array([[lon_deg[int(i/dt)], lat_deg[int(i/dt)], cap[int(i/dt)]]]).T
    Xhat, P, ytilde, innov_norm = Kalman(Xhat, P, u, Y, Q, R, Fk, Gk, Hk)


    #Display the results if needed
    if display_bot:
        draw_tank(Xhat)
        # draw_ellipse_cov(ax, Xhat[0:2], 10*P[0:2, 0:2], 0.9, col='black')
        ax.scatter(Xhat[0, 0], Xhat[1, 0], color='red', label = 'Estimation of position', s = 5)
        ax.legend()
        pause(0.001)
        clear(ax)
        legende(ax)
        

    #Appending lists to visualize our covariance model
    T.append(i)
    i = int(i/dt)
    for j in range(P.shape[0]):
        for k in range(P.shape[1]):
            c = P.shape[0]
            PMatrix[i, j+c*k] = P[j, k]
    
    for j in range(Y.shape[0]):
        for k in range(Y.shape[1]):
            c = Y.shape[0]
            Innov[i, j+c*k] = ytilde[j, k]
            Innov_norm[i, j+c*k] = innov_norm[j, k]


if __name__ == "__main__":
    T = [int(i/dt)/3600 for i in T]

    #Ploting some useful contents
    plt.close()
    plt.figure()
    plt.suptitle(f"P(t) Matrix : {N} epoch with step dt={dt}")
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            ax = plt.subplot2grid((P.shape[0], P.shape[1]), (i, j))
            c = P.shape[0]
            ax.plot(T, PMatrix[:, i+c*j])
            ax.set_xlabel("Time [h]")
            ax.set_ylabel("Error [m]")
            ax.set_title(f"P_{i},{j}")
    plt.show()

    print("P : ", P)

    plt.figure()
    plt.suptitle(f"Innovation Matrix : {N} epoch with step dt={dt}")
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            ax = plt.subplot2grid((Y.shape[0], 2*Y.shape[1]), (i, j))
            c = Y.shape[0]
            ax.plot(T, Innov[:, i+c*j])
            ax.set_xlabel("Time [h]")
            ax.set_ylabel("Innovation [m]")
            ax.set_title(f"Innov_{i},{j}")

            if j==0:
                ax = plt.subplot2grid((Y.shape[0], 2*Y.shape[1]), (i, 1))
            else:
                 ax = plt.subplot2grid((Y.shape[0], 2*Y.shape[1]), (i, 2*j))               
            c = Y.shape[0]
            ax.plot(T, Innov_norm[:, i+c*j])
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Innovation normalisée[m]")
            ax.set_title(f"Innov_norm_{i},{j}")
    plt.show()

    if save_data == True:
        np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)) + "\log", "innovation_normalisee.npz"), innovation_normalisee=Innov_norm)
