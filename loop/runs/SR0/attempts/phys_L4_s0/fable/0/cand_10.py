def equation(t, x, v, params):
    p=params
    return np.exp(-p[2]*t)*(p[0]*x+p[9])+p[3]*x**3+p[4]*x+p[5]*t*np.tanh(p[6]*v)+p[7]*np.abs(v)**p[1]+p[8]*v