def equation(t, x, v, params):
    p=params
    return p[0]+p[1]*x+p[2]*x**3+p[3]*v+p[4]*np.abs(x)**p[5]*np.exp(-p[6]*t)+p[7]*t*np.tanh(p[8]*v)+p[9]*v*np.log(v)