def equation(t, A, params):
    return 10*params[0]*np.exp(-0.01*params[1]*t**2) - 0.1*params[2]*A**3 - 0.3*params[3]*A**2*t*np.exp(-0.3*params[4]*t)