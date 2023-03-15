from roblib import *


def f(x, u):
    x_dot = np.array([[x[3, 0]],
                      [x[4, 0]],
                      [u[0, 0]],
                      [u[1, 0]],
                      [u[2, 0]]])
    return x_dot


def Kalman(xbar, P, u, y, Q, R, F, G, H):
    # Prédiction
    xbar = F @ xbar + G @ u
    P = F @ P @ F.T + Q

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
ax = init_figure(-60, 60, -60, 60)

P = 0.01 * np.eye(5)
X = np.array([[10], [10], [0], [0], [0]])
Xhat = X
u = np.array([[1], [0.5], [0]])

sigm = 0.0001
Q = np.diag([sigm, sigm, sigm, sigm, sigm])
R = np.diag([sigm])

for i in np.arange(0, 60*dt, dt):
    clear(ax)
    legende(ax)
    draw_tank(X)

    Hk = np.array([[1, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0]])

    Fk = np.eye(5)
    # Fk = np.eye(5) + dt * np.array([[0, 0, 0, np.cos(X[2, 0]), np.sin(X[2, 0])],
    #                                 [0, 0, 0, np.sin(X[2, 0]), -np.cos(X[2, 0])],
    #                                 [0, 0, 0, 0, 0],
    #                                 [0, 0, 0, 0, 0],
    #                                 [0, 0, 0, 0, 0]])

    Gk = dt * np.array([[0, 0, 0],
                        [0, 0, 0],
                        [1, 0, 0],
                        [0, np.cos(X[2, 0]), np.sin(X[2, 0])],
                        [0, np.sin(X[2, 0]), -np.cos(X[2, 0])]])
    # Gk = dt * np.array([[0, 0, 0],
    #                     [0, 0, 0],
    #                     [1, 0, 0],
    #                     [0, 1, 0],
    #                     [0, 0, 1]]) 


    # Sans bruits
    y = np.array([[X[0, 0]], [X[1, 0]]])
    Xhat, P, ytilde = Kalman(Xhat, P, u, y, Q, R, Fk, Gk, Hk)

    # Display the results
    draw_ellipse_cov(ax, Xhat[0:2], P[0:2, 0:2], 0.9, Xhat[2,0], col='black')
    plt.scatter(Xhat[0, 0], Xhat[1, 0], color='red', label = 'Estimation of position', s = 5)

    plt.legend()

    pause(.001)

    #Avancement de l'état vrai
    X = X + dt*f(X,u)