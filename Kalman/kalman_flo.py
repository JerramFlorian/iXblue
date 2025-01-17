import os
from roblib import *
import sys
from tqdm import tqdm
import qrunch



#Importing the data (lat, lon, heading)
file_path = os.path.dirname(os.path.abspath(__file__)) + "\log"
allan = np.load(file_path + "\deviation_allan.npz")
dev_lat, dev_lon, dev_cap, dev_lat_m, dev_lon_m = np.sqrt(3)*10**(-8.125), np.sqrt(3)*10**(-7.731), 0.0825, np.sqrt(3)*10**(-2.931), np.sqrt(3)*10**(-2.568)
# dev_lat, dev_lon, dev_cap = np.sqrt(3)*10**(-8.9), np.sqrt(3)*10**(-8.9), 0.07


def f(X, u):
    u1, u2, u3 = u.flatten()
    x1, x2, x3, x4, x5 = X.flatten()
    x_dot = np.array([[x1 + dt*x4],
                      [x2 + dt*x5],
                      [x3 + dt*u1],
                      [x4 + dt*(u2*np.sin(x3) - u3*np.cos(x3))],
                      [x5 + dt*(u2*np.cos(x3) + u3*np.sin(x3))]])
    return x_dot

def Kalman(xbar, P, u, y, Q, R, F, G, H, gnss=True):
    # Prédiction
    xbar = f(xbar, u)
    P = F @ P @ F.T + G @ Q @ G.T

    if gnss == False:
        return(xbar, P)

    # Correction
    ytilde = y - (H @ xbar)
    # print("Hx : ", H @ xbar)
    # print("ytilde : ", ytilde)
    S = H @ P @ H.T + R
    innov_norm = sqrtm(np.linalg.inv(S))@ytilde
    K = P @ H.T @ np.linalg.inv(S)
    # print('K : ', K)
    # print('Ky : ', K@ytilde)
    xbar = xbar + K @ ytilde
    # print('Corriged xbar : ', xbar)
    P = P - K @ H @ P

    return xbar, P, ytilde, innov_norm

def control(x,t):
    x1, x2, x3, x4, x5 = x.flatten()
    L = 15
    K = 0.5
    f = 0.1
    nb = 2 

    w=L*array([[cos(f*t)],[sin(f*nb*t)]])  
    dw=L*array([[-f*sin(f*t)],[nb*f*cos(f*nb*t)]])  
    ddw=L*array([[-f**2*cos(f*t)],[-nb**2*f**2*sin(f*nb*t)]])
    
    Ax = np.array([[np.sin(x3), -np.cos(x3)],
                [np.cos(x3), np.sin(x3)]])
    
    y = array([[x1],
               [x2]])
    
    dy = array([[x4],
                [x5]])
    
    ddy = w-y + 2*(dw-dy) + ddw
    
    u = inv(Ax)@ddy
    u1 = -K*sawtooth(x3 - np.arctan2(w[1,0]-x2,w[0,0] - x1))
    u = np.vstack((u1,u))
    if np.linalg.norm(u) > 500:
        sys.exit()

    plt.scatter(w[0,0],w[1,0], color = 'red', label = 'point to follow')
    W = lambda f : np.array([L*np.cos(f*t), L*np.sin(nb*f*t)])
    plt.scatter(W(np.linspace(0,100,1100))[0],W(np.linspace(0,100,1100))[1], s = 0.1, label='trajectory to follow')

    return u 

#Génération du rw
def noise_sensor(n, sigma_lon, sigma_lat):
    rw = np.zeros((3, n))
    for i in range(1, n):
        rw[0, i] = rw[0, i-1] + np.random.normal(scale=sigma_lon)
        rw[1, i] = rw[1, i-1] + np.random.normal(scale=sigma_lat)
    return rw
    
def draw_ellipse0(ax, c, Γ, a, col,coledge='black'):  # classical ellipse (x-c)T * invΓ * (x-c) <a^2
    # draw_ellipse0(ax,array([[1],[2]]),eye(2),a,[0.5,0.6,0.7])
    A = a * sqrtm(Γ)
    w, v = eig(A)
    v1 = np.array([[v[0, 0]], [v[1, 0]]])
    v2 = np.array([[v[0, 1]], [v[1, 1]]])
    f1 = A @ v1
    f2 = A @ v2
    φ = (arctan2(v1[1, 0], v1[0, 0]))
    α = φ * 180 / 3.14
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
    # ax.set_xlim(X[0,0]-40, X[0,0]+40)
    # ax.set_ylim(X[1,0]-40, X[1,0]+40)
    ax.set_title('Filtre de Kalman')
    ax.set_xlabel('x')
    ax.set_ylabel('y')


dt = 0.1
display_bot = False
fullGNSS = True
if display_bot:
    ax = init_figure(-20, 20, -20, 20)

P = 0.01 * np.eye(5)
X = np.array([[0], [0], [0], [0], [0]])
Xhat = X
Y = np.zeros((3, 1))

sigm_equation = 0.1
sigm_measure = 0.001
Q = np.diag([sigm_equation, sigm_equation, sigm_equation]) #matrice de covariance de bruit pour l'état du système
R = np.diag([dev_lon_m**2, dev_lat_m**2, dev_cap**2]) #matrice de covariance de bruit pour les mesures
# R = np.diag([sigm_measure, sigm_measure, sigm_measure])

u = np.array([[0], [0], [0]])

T = []
N = 5000
PMatrix = np.zeros((N, P.shape[0]*P.shape[1]))
Innov = np.zeros((N, Y.shape[0]*Y.shape[1]))
Innov_norm = np.zeros((N, Y.shape[0]*Y.shape[1]))


for i in tqdm(np.arange(0, N*dt, dt)):
    # Real state
    u = control(X,i)
    X = f(X,u)

    u1, u2, u3 = u.flatten()
    x1, x2, x3, x4, x5 = X.flatten()

    Hk = np.array([[1, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0],
                   [0, 0, 1, 0, 0]])

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

    if fullGNSS: #or i%1==0: #each second we get a new value of GNSS data
        BR1 = noise_sensor(int(i/dt)+1, dev_lon_m, dev_lat_m)
        BR = BR1[:,-1].reshape(3, 1)
        # print("BR : ", BR)
        Y = np.array([[x1, x2, x3]]).T + BR
        # print("Y : ", Y)
        Xhat, P, ytilde, innov_norm = Kalman(Xhat, P, u, Y, Q, R, Fk, Gk, Hk)
    else:
        Xhat, P = Kalman(Xhat, P, u, Y, Q, R, Fk, Gk, Hk, False)


    if display_bot:
        # Display the results
        draw_tank(X)
        draw_ellipse_cov(ax, Xhat[0:2], 10**4*P[0:2, 0:2], 0.9, col='black')
        ax.scatter(Xhat[0, 0], Xhat[1, 0], color='red', label = 'Estimation of position', s = 5)
        ax.legend()
        pause(0.001)
        clear(ax)
        legende(ax)
        

    # Append lists to visualize our covariance model
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
    plt.close()
    plt.figure()
    plt.suptitle(f"P(t) Matrix : {N} epoch with step dt={dt}")
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            ax = plt.subplot2grid((P.shape[0], P.shape[1]), (i, j))
            c = P.shape[0]
            ax.plot(T, PMatrix[:, i+c*j])
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Error [m/rad]")
            ax.set_title(f"P_{i},{j}")
    # plt.show()

    T_allan, dev_allan, _ = qrunch.allan_deviation(BR1[0, :])
    a_th, b_th = np.polyfit(np.log10(T_allan), np.log10(dev_allan), 1)
    plt.figure()
    plt.loglog(T_allan, dev_allan, label=f" : {'%.4g'%a_th}t + {'%.4g'%b_th}")
    plt.xlabel("Time [s]")
    plt.ylabel("Bruit [m]")
    plt.title("Log du bruit")
    plt.legend()
    plt.show()

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
    plt.show()


    # ax1 = plt.subplot2grid((1, 5), (0, 0))
    # ax2 = plt.subplot2grid((1, 5), (0, 1))
    # ax3 = plt.subplot2grid((1, 5), (0, 2))
    # ax4 = plt.subplot2grid((1, 5), (0, 3))
    # ax5 = plt.subplot2grid((1, 5), (0, 4))

    # ax1.plot(T,P11)
    # ax1.set_title('P11 (x)')
    # ax1.set_xlabel('time [s]')
    # ax1.set_ylabel('error [rad]')

    # ax2.plot(T,P22)
    # ax2.set_title('P22 (y)')
    # ax2.set_xlabel('time [s]')
    # ax2.set_ylabel('error [m]')

    # ax3.plot(T,P33)
    # ax3.set_title('P33 (theta)')
    # ax3.set_xlabel('time [s]')
    # ax3.set_ylabel('error [m]')

    # ax4.plot(T,P44)
    # ax4.set_title('P44 (vx)')
    # ax4.set_xlabel('time [s]')
    # ax4.set_ylabel('error [m]')

    # ax5.plot(T,P55)
    # ax5.set_title('P55 (vy)')
    # ax5.set_xlabel('time [s]')
    # ax5.set_ylabel('error [m]')

    # plt.show()