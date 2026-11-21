def equation(eps, T, params):
    p = params
    return p[0]*eps*np.exp(-p[1]*eps)/T + p[2] + p[3]*np.abs(T)**p[4] + p[5]*np.exp(-10*p[6]*(1/np.abs(T)**p[7] - 0.2))