from roblib import *
import os
from tqdm import tqdm

file_path = os.path.dirname(os.path.abspath(__file__))
file_path = file_path+"\log\gnss_test_station.nma"
trames = np.genfromtxt(file_path, delimiter = ',', dtype = 'str', comments = '$GPZDA', skip_footer = 1)
lat = trames[:, 2]
lon = trames[:, 4]
lat_deg = ['' for i in range(len(lat))]
lon_deg = ['' for i in range(len(lon))]
for i in range(len(lat)):
    lat_deg[i] += lat[i][0:2] + '.' + str(np.float64(lat[i][2:])/60)[2:8]
    lat_deg[i] = np.float64(lat_deg[i])
    lon_deg[i] += lon[i][0:3] + '.' + str(np.float64(lon[i][3:])/60)[2:8]
    lon_deg[i] = np.float64(lon_deg[i])


def f(X, u):
    u1, u2, u3 = u.flatten()
    x1, x2, x3, x4, x5 = X.flatten()
    x_dot = np.array([[x1 + dt*x4],
                      [x2 + dt*x5],
                      [x3 + dt*u1],
                      [x4 + dt*(u2*np.cos(x3) + u3*np.sin(x3))],
                      [x5 + dt*(u2*np.sin(x3) - u3*np.cos(x3))]])
    return x_dot

def Kalman(xbar, P, u, y, Q, R, F, G, H, gnss=True):
    # Prédiction
    xbar = f(xbar, u) #+ mvnrnd1(Gk @ Q @ Gk.T) #+ bruit(xbar, 0, 0.1)
    P = F @ P @ F.T + G @ Q @ G.T

    if gnss == False:
        return(xbar, P)

    # Correction
    ytilde = y - (H @ xbar) #+ bruit(H@xbar, 0, 0.1))
    # print("Innovation : ", ytilde[0, 0], " ; ", ytilde[1, 0], " ; ", ytilde[2, 0])
    S = H @ P @ H.T + R
    innov_norm = sqrtm(np.linalg.inv(S))@ytilde
    # print("Innovation normalisée : ", innov_norm[0, 0], " ; ", innov_norm[1, 0], " ; ", innov_norm[2, 0])
    K = P @ H.T @ np.linalg.inv(S)
    xbar = xbar + K @ ytilde
    P = P - K @ H @ P

    return xbar, P, ytilde, innov_norm

def bruit(M, mean, std):
    B = np.random.normal(mean, std, size=(M.shape[0], M.shape[1]))
    # print('bruit : ', B)
    return(B)

def draw_ellipse(c, Γ, η, theta, ax, col):
    if norm(Γ) == 0:
        Γ = Γ + 0.001 * np.eye(len(Γ[1,:]))
    a = np.sqrt(-2 * np.log(1 - η))
    w, v = np.linalg.eigh(Γ)
    idx = w.argsort()[::-1]
    w, v = w[idx], v[:, idx]
    v1 = np.array([[v[0, 0]], [v[1, 0]]])
    v2 = np.array([[v[0, 1]], [v[1, 1]]])
    f1 = a * np.sqrt(w[0]) * v1
    f2 = a * np.sqrt(w[1]) * v2
    e = Ellipse(xy=c, width=2 * norm(f1), height=2 * norm(f2), angle=theta*180/pi)
    ax.add_artist(e)
    e.set_clip_box(ax.bbox)
    e.set_alpha(0.7)
    e.set_facecolor('none')
    e.set_edgecolor(col)
def draw_ellipse0(ax, c, Γ, a, α, col,coledge='black'):  # classical ellipse (x-c)T * invΓ * (x-c) <a^2
    # draw_ellipse0(ax,array([[1],[2]]),eye(2),a,[0.5,0.6,0.7])
    A = a * sqrtm(Γ)
    w, v = eig(A)
    v1 = np.array([[v[0, 0]], [v[1, 0]]])
    v2 = np.array([[v[0, 1]], [v[1, 1]]])
    f1 = A @ v1
    f2 = A @ v2
    # φ = (arctan2(v1[1, 0], v1[0, 0]))
    # α = φ * 180 / 3.14
    e = Ellipse(xy=c, width=2 * norm(f1), height=2 * norm(f2), angle=α*180/np.pi)
    ax.add_artist(e)
    e.set_clip_box(ax.bbox)

    e.set_alpha(0.7)
    e.set_facecolor(col)
    e.set_edgecolor(coledge)

    # e.set_fill(False)
    # e.set_alpha(1)
    # e.set_edgecolor(col)
def draw_ellipse_cov(ax,c,Γ,η, α=0, col ='blue',coledge='black'): # Gaussian confidence ellipse with artist
    #draw_ellipse_cov(ax,array([[1],[2]]),eye(2),0.9,[0.5,0.6,0.7])
    if (np.linalg.norm(Γ)==0):
        Γ=Γ+0.001*eye(len(Γ[1,:]))
    a=np.sqrt(-2*log(1-η))
    draw_ellipse0(ax, c, Γ, a, α,col,coledge)

def legende(ax):
    # ax.set_xlim(X[0,0]-40, X[0,0]+40)
    # ax.set_ylim(X[1,0]-40, X[1,0]+40)
    ax.set_title('Filtre de Kalman')
    ax.set_xlabel('x')
    ax.set_ylabel('y')


dt = 0.1
display_bot = False
# fullGNSS = False
if display_bot:
    # ax = init_figure(-50, 50, -50, 50)
    ax = init_figure(40, 55, -5, 10)

P = 0.01 * np.eye(5)
Xhat = np.array([[lat_deg[0]], [lon_deg[0]], [0], [0], [0]])
Y = np.zeros((3, 1))

sigm_equation = 0.001
sigm_measure = 0.001
Q = np.diag([sigm_equation, sigm_equation, sigm_equation])
R = np.diag([5*sigm_measure, sigm_measure])

u = np.array([[0], [0], [0]])

T = []
N = len(lat_deg)
PMatrix = np.zeros((N, P.shape[0]*P.shape[1]))
Innov = np.zeros((N, Y.shape[0]*Y.shape[1]))
Innov_norm = np.zeros((N, Y.shape[0]*Y.shape[1]))


for i in tqdm(np.arange(0, N*dt, dt)):
    # Real state

    u1, u2, u3 = u.flatten()
    x1, x2, x3, x4, x5 = Xhat.flatten()

    Hk = np.array([[1, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0]])

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


    Y = np.array([[lat_deg[int(i/dt)], lon_deg[int(i/dt)]]]).T #+ mvnrnd1(R)
    Xhat, P, ytilde, innov_norm = Kalman(Xhat, P, u, Y, Q, R, Fk, Gk, Hk)


    if display_bot:
        # Display the results
        draw_tank(Xhat)
        draw_ellipse_cov(ax, Xhat[0:2], 10*P[0:2, 0:2], 0.9, Xhat[2,0], col='black')
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
            ax.set_xlabel("Time [s]")
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


    # ax1 = plt.subplot2grid((1, 5), (0, 0))
    # ax2 = plt.subplot2grid((1, 5), (0, 1))
    # ax3 = plt.subplot2grid((1, 5), (0, 2))
    # ax4 = plt.subplot2grid((1, 5), (0, 3))
    # ax5 = plt.subplot2grid((1, 5), (0, 4))

    # ax1.plot(T,P11)
    # ax1.set_title('P11 (x)')
    # ax1.set_xlabel('time [s]')
    # ax1.set_ylabel('error [m]')

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