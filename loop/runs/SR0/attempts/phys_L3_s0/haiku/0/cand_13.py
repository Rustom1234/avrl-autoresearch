def equation(t, x, v, params):
    p = params
    return (p[0] + p[1]*x + p[2]*v + p[3]*x**3
            + p[4]*t + p[5]*t*x + p[6]*t*x*x + p[7]*t*x**3
            + p[8]*t*t + p[9]*t*v)
