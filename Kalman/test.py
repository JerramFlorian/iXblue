from roblib import *  # available at https://www.ensta-bretagne.fr/jaulin/roblib.py
from numpy import *


def f(x,u):
    x,u=x.flatten(),u.flatten()
    xdot = array([[x[3]*cos(x[2])],[x[3]*sin(x[2])],[u[0]],[u[1]]])
    return(xdot)

def control(x,w,dw,ddw):
    x1, x2, x3, x4 = x.flatten()
    
    Ax = array([[-x4*sin(x3), cos(x3)],
                [x4*cos(x3), sin(x3)]])
    y = array([[x1],
               [x2]])
    dy = array([[x4*cos(x3)],
                [x4*sin(x3)]])
    ddy = w-y + 2*(dw-dy) + ddw
    u = inv(Ax)@ddy
    return u    

def calcul_w(t,i):
    s = a*t+2*i*pi/m
    c = L*array([[cos(s)],[sin(s)]]) 
    dc = L*a*array([[-sin(s)],[cos(s)]]) 
    ddc = -L*a**2*c
    D = diag([20 + 15*sin(a*t), 20])
    dD = diag([a*15*cos(a*t), 0])
    ddD = diag([-a**2*15*sin(a*t), 0])
    R = array([[cos(a*t), -sin(a*t)],
               [sin(a*t), cos(a*t)]])
    dR = a*array([[-sin(a*t), -cos(a*t)],
               [cos(a*t), -sin(a*t)]])
    ddR = -a**2*R
    w = R@D@c
    dw = R@D@dc + R@dD@c + dR@D@c
    ddw = R@D@ddc + R@ddD@c + ddR@D@c + 2*(R@dD@dc + dR@D@dc + dR@dD@c)
    
    return w, dw ,ddw

L = 1
r = 0.05
d = 50*r
m   = 20
X   = 10*randn(4,m)
a,dt = 0.1,0.1

ax=init_figure(-d,d,-d,d)

for t in arange(0,10,dt):
    clear(ax)
    for i in range(m): 
        s = a*t+2*i*pi/m
        w = L*array([[cos(s)],[sin(s)]]) 
        dw = L*a*array([[-sin(s)],[cos(s)]]) 
        ddw = -L*a**2*w 
        x=X[:,i].reshape(4,1)
        u = control(x,w,dw,ddw)
        x=X[:,i].reshape(4,1)
        draw_tank(x,'b',r)
        x=x+f(x,u)*dt        
        X[:,i]  = x.flatten()
        plot([w[0][0]],[w[1][0]],'r+')

pause(1)
L = 1
d = 50
ax=init_figure(-d,d,-d,d)
X   = 10*randn(4,m)
for t in arange(0,30,dt):
    clear(ax)
    for i in range(m): 
        w, dw, ddw = calcul_w(t, i)
        x=X[:,i].reshape(4,1)
        u = control(x,w,dw,ddw)
        x=X[:,i].reshape(4,1)
        draw_tank(x,'b')
        x=x+f(x,u)*dt        
        X[:,i]  = x.flatten()
        plot([w[0][0]],[w[1][0]],'r+')

