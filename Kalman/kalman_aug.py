from roblib import *
import sys

global dt
dt = 0.1

def f(X, u):
    
    u1, u2, u3 = u.flatten()
    x, y, θ, vx, vy = X.flatten()
    x_dot = np.array([[vx],
                      [vy],
                      [u1],
                      [u2*np.cos(θ) + u3*np.sin(θ)],
                      [u2*np.sin(θ) - u3*np.cos(θ)]])
    return x_dot

sawtooth = lambda x : (2*np.arctan(np.tan(x/2)))

def control(x,t):
    x1, x2, x3, x4, x5 = x.flatten()
    L = 15
    K = 0.5
    f = 0.1
    nb = 2 

    w=L*array([[cos(f*t)],[sin(f*nb*t)]])  
    dw=L*array([[-f*sin(f*t)],[nb*f*cos(f*nb*t)]])  
    ddw=L*array([[-f**2*cos(f*t)],[-nb**2*f**2*sin(f*nb*t)]])
    
    Ax = array([[cos(x3), sin(x3)],
                [sin(x3), -cos(x3)]])
    
    y = array([[x1],
               [x2]])
    
    dy = array([[x4],
                [x5]])
    
    ddy = w-y + 2*(dw-dy) #+ ddw
    
    u = inv(Ax)@ddy
    u1 = -K*sawtooth(x3 - np.arctan2(w[1,0]-x2,w[0,0] - x1))
    print(u)
    u = np.vstack((u1,u))

    print(u,'\n')
    if np.linalg.norm(u) > 500:
        sys.exit()

    scatter(w[0,0],w[1,0], color = 'red', label = 'point to follow')
    W = lambda f : np.array([L*cos(f*t), L*sin(nb*f*t)])
    scatter(W(np.linspace(0,100,10000))[0],W(np.linspace(0,100,10000))[1], s = 0.1)

    return u 

# List of waypoints

Wps = 10*array([[0],#,0,15,30,15],
              [1]])#,25,30,15,20]])

# Wps = array([[0,15,30,15],
#               [25,30,15,20]])

# a = -20
# b = 20
# N = 5
# Wps = np.random.uniform(low=a, high=b, size=(2, N))

# Observation function
def g(x):
    x=x.flatten()
    wp_detected = False
    H = array([[0,0,0,0,0]])
    y = array([[0]])
    Beta = [] 
    for i in range(Wps.shape[1]):
        a=Wps[:,i].flatten() #wps(i) in (xi,yi)
        da = a-(x[0:2]).flatten()
        dist = norm(da)      
        if dist < 15:
            wp_detected = True
            plot(array([a[0],x[0]]),array([a[1],x[1]]),"red",1)
            δ = arctan2(da[1],da[0])
            Hi = array([[-sin(δ),cos(δ), 0, 0, 0]])
            yi = [[-sin(δ)*a[0] + cos(δ)*a[1]]]
            if np.linalg.norm(H) == 0:
                H = Hi; y = yi
            else:
                H = vstack((H,Hi)); y = vstack((y,yi))     
                
            Beta.append(0.0001)
    Γβ = diag(Beta)
    if len(Beta) != 0:
        y = y + mvnrnd1(Γβ)
    return H, y, Γβ, wp_detected

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

if __name__ == "__main__":
    dt = 0.1
    display_bot = True
    UWB = True
    GNSS = False

    if display_bot:
        ax = init_figure(-20, 20, -20, 20)
    T = []

    import sys
    N = 1000
    PMatrix = np.zeros((N,25))

    P = 10 * np.eye(5)
    X = np.array([[0], [0], [0], [0], [0]])
    Xhat = X

    sigm_equation = 0.1
    sigm_measure = 0.1
    Q = np.diag([sigm_equation, sigm_equation, sigm_equation])
    R = np.diag([sigm_measure, sigm_measure])

    # u = (w, ax, ay)
    u = np.array([[0.1], [0.2], [-1]])

    for i in np.arange(0, N*dt, dt):
        
        # Real state
        u = control(X,i)
        X = X + dt*f(X,u)
        
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

        if GNSS:
            if i%1==0: #each second we get a new value of GNSS data
                Y = np.array([[x], [y]]) + mvnrnd1(R)
                Hk = np.array([[1, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0]])
                Xhat, P, ytilde = Kalman(Xhat, P, u, Y, Q, R, Fk, Gk, Hk)
            else:
                Xhat = Xhat + dt*f(Xhat,u) + mvnrnd1(Gk @ Q @ Gk.T) #Fk @ Xhat + Gk @ u
                P = Fk @ P @ Fk.T + Gk @ Q @ Gk.T

        if UWB :
            Hk,Y,R,wp_detected = g(X)
            if wp_detected:
                Xhat, P, ytilde = Kalman(Xhat, P, u, Y, Q, R, Fk, Gk, Hk)
            else:
                Xhat = Xhat + dt*f(Xhat,u) + mvnrnd1(Gk @ Q @ Gk.T) #Fk @ Xhat + Gk @ u
                P = Fk @ P @ Fk.T + Gk @ Q @ Gk.T

        if display_bot:
            # Display the results
            draw_tank(X)
            draw_ellipse_cov(ax, Xhat[0:2], P[0:2, 0:2], 0.9, Xhat[2,0], col='black')
            ax.scatter(Xhat[0, 0], Xhat[1, 0], color='red', label = 'Estimation of position', s = 5)
            ax.legend()
            # ax.set_xlim(X[0,0]-21, X[0,0]+21)
            # ax.set_ylim(X[1,0]-21, X[1,0]+21)

            scatter(Wps[0], Wps[1])

            pause(0.001)

            clear(ax)
            legende(ax)

        # Append lists to visualize our covariance model
        T.append(i)
        i = int(i/dt)
        for j in range(5):
            for k in range(5):
                PMatrix[i,j+5*k] = P[j,k]


    plt.close()
    plt.figure()
    plt.suptitle(f"P(t) Matrix")
    AX = []
    for i in range(5):
        for j in range(5):
            ax = plt.subplot2grid((5, 5), (i, j))
            ax.scatter(T,PMatrix[:,i+5*j],color='darkblue',s=1)
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Error [m]")
            ax.set_title(f"P_{i},{j}")
    plt.show()