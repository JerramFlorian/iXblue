from roblib import *


def f(x, u):
    x_dot = np.array([[x[3, 0]],
                      [x[4, 0]],
                      [u[0, 0]],
                      [u[1, 0]*np.cos(x[2, 0]) + u[2, 0]*np.sin(x[2, 0])],
                      [u[1, 0]*np.sin(x[2, 0]) - u[2, 0]*np.cos(x[2, 0])]])
    return x_dot


def Kalman(xbar, P, u, y, Q, R, F, G, H):
    # Prédiction
    xbar = xbar + dt*f(xbar,u) #F @ xbar + G @ u
    P = F @ P @ F.T + G @ Q @ G.T

    # Correction
    ytilde = y - H @ xbar
    Γy = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(Γy)
    xbar = xbar + K @ ytilde
    P = P - K @ H @ P

    return xbar, P, ytilde

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
    ax.set_title('Filtre de Kalman')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

dt = 0.1
display_bot = False
fullGNSS = False
if display_bot:
    ax = init_figure(-60, 60, -60, 60)
T = []

import sys
N = np.arange(0, 100*dt, dt).shape[0]
PMatrix = np.zeros((N,25))

P = 0.01 * np.eye(5)
X = np.array([[0], [0], [0], [0], [0]])
Xhat = X

sigm_equation = 0
sigm_measure = 0.0001
Q = np.diag([sigm_equation, sigm_equation, sigm_equation])
R = np.diag([5*sigm_measure, sigm_measure])


u = np.array([[1], [0.1], [1]])

for i in np.arange(0, 100*dt, dt):

    # Real state
    X = X + dt*f(X,u)

    Hk = np.array([[1, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0]])

    u1, u2, u3 = u.flatten()
    x, y, θ, vx, vy = X.flatten()
    Fk = np.eye(5) + dt * np.array([[0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 1],
                                    [0, 0, -u2*np.sin(θ) + u3*np.cos(θ), 0, 0],
                                    [0, 0, u2*np.cos(θ) + u3*np.sin(θ), 0, 0],
                                    [0, 0, 0, 0, 0]])

    Gk = dt * np.array([[0, 0, 0],
                        [0, 0, 0],
                        [1, 0, 0],
                        [0, np.cos(θ), np.sin(θ)],
                        [0, np.sin(θ), -np.cos(θ)]])

    if fullGNSS or i%1==0: #each second we get a new value of GNSS data
        Y = np.array([[x], [y]])
        Xhat, P, ytilde = Kalman(Xhat, P, u, Y, Q, R, Fk, Gk, Hk)
    else:
        Xhat = Xhat + dt*f(Xhat,u) #Fk @ Xhat + Gk @ u
        P = Fk @ P @ Fk.T + Gk @ Q @ Gk.T

    if display_bot:
        # Display the results

        clear(ax)
        legende(ax)

        draw_tank(X)
        draw_ellipse_cov(ax, Xhat[0:2], 10000*P[0:2, 0:2], 0.9, Xhat[2,0], col='black')
        ax.scatter(Xhat[0, 0], Xhat[1, 0], color='red', label = 'Estimation of position', s = 5)
        ax.legend()
        ax.set_xlim(X[0,0]-50, X[0,0]+50)
        ax.set_ylim(X[1,0]-50, X[1,0]+50)

        pause(0.001)

    # Append lists to visualize our covariance model
    T.append(i)
    i = int(i/dt)
    print(i)
    for j in range(5):
        for k in range(5):
            print(i,j+5*k)
            PMatrix[i,j+5*k] = P[j,k]

if __name__ == "__main__":
    plt.close()
    plt.figure()
    plt.suptitle(f"P(t) Matrix")
    AX = []
    for i in range(5):
        for j in range(5):
            ax = plt.subplot2grid((5, 5), (i, j))
            ax.plot(T,PMatrix[:,i+5*j])
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Error [m]")
            ax.set_title(f"P_{i},{j}")
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