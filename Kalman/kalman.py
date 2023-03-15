from roblib import *


def f(x, u):
    x_dot = np.array([[x[3, 0]],
                      [x[4, 0]],
                      [u[0, 0]],
                      [u[1, 0]],
                      [u[2, 0]]])
    return x_dot

def Kalman(xbar, P, u, y, Pa, Pb, A, B, C): #ajouter ytilde, l'innovation
    ytilde = y - C @ xbar
    print(ytilde)
    Γy = C @ P @ C.T + Pb
    # print('Gy : ', Γy)
    # print('P : ', P)
    K = P @ C.T @ np.linalg.inv(Γy)
    xup = xbar + K @ ytilde
    Γup = P - K @ C @ P
    Xhat = A @ xup + B @ u
    Γ1 = A @ Γup @ A.T + Pa
    return Xhat, Γ1

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
    φ = np.arctan2(v1[1, 0].real, v1[0, 0].real)
    α = φ * 180 / np.pi
    # e = Ellipse(xy=c, width=2 * norm(f1), height=2 * norm(f2), angle=α)
    e = Ellipse(xy=c, width=2 * norm(f1), height=2 * norm(f2), angle=theta*180/pi)
    ax.add_artist(e)
    e.set_clip_box(ax.bbox)
    e.set_alpha(0.7)
    e.set_facecolor('none')
    e.set_edgecolor(col)

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

sigm = 0.000001
Pa = np.diag([sigm, sigm, sigm, sigm, sigm])
Pb = np.diag([10])


for i in np.arange(0, 300*dt, dt):
    clear(ax)
    legende(ax)
    draw_tank(X)

    Hk = np.array([[1, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0]])

    Fk = np.eye(5) + dt * np.array([[0, 0, 0, np.cos(X[2, 0]), np.sin(X[2, 0])],
                                    [0, 0, 0, np.sin(X[2, 0]), -np.cos(X[2, 0])],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0]])

    Gk = dt * np.array([[0, 0, 0],
                        [0, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]]) #inverser la matrice entre F et G

    # Sans bruits
    y = np.array([[X[0, 0]], [X[1, 0]]])
    Xhat, P = Kalman(Xhat, P, u, y, Pa, Pb, Fk, Gk, Hk)
    
    draw_ellipse(Xhat[0:2], 10000*P[0:2, 0:2], 0.9, Xhat[2,0], ax, col='black')
    plt.scatter(Xhat[0, 0], Xhat[1, 0], color='red', label = 'Estimation of position', s = 5)

    plt.legend()

    pause(.001)

    #Avancement de l'état vrai
    X = X + dt*f(X,u)