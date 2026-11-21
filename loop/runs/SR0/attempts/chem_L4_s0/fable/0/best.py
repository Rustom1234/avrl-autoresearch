def equation(t, A, params):
    p=params
    u=0.1*np.abs(p[3])*t**(2.3*p[4])*A**(1.44*p[5])
    return 7.2*p[0]+6.5*p[1]*np.log(A)+8*p[2]*u/(1+u)-6.8*p[6]*A-0.27*p[7]*A*t+0.25*p[8]*t+1.14*p[9]*A**2