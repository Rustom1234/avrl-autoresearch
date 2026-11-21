def equation(t, x, v, params):
    p=params
    return p[0]*x+p[1]*v+p[2]*x**3+p[3]*np.abs(x)**p[4]+p[5]*v**p[6]+p[7]*x*v+p[8]*x*x*v+p[9]