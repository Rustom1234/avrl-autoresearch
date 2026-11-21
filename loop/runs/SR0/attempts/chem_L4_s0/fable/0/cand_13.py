def equation(t, A, params):
    p=params
    u=0.17*np.abs(p[3])*t**(1.3*p[4])*A**(0.76*p[5])
    return 6.7*p[0]+6.6*p[1]*np.log(A)+7.6*p[2]*np.tanh(u)-6.8*p[6]*A-0.25*p[7]*A*t+0.27*p[8]*t+1.12*p[9]*A**2