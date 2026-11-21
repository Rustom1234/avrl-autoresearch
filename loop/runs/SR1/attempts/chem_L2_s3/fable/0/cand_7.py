def equation(t, A, params):
    p=params
    return -p[0]*A**2*np.sqrt(t) - p[1]/t - p[2]*t/(1+p[3]*t)