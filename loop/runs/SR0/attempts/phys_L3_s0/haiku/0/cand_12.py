def equation(t, x, v, params):
    p = params
    return (p[0] + p[1]*x + p[2]*v + p[3]*x**3 + p[4]*v*v
            + p[5]*t + p[6]*t*x + p[7]*t*v + p[8]*t*x*x + p[9]*t*v*v
            + p[10]*t*x**3 + p[11]*t*t + p[12]*t*t*x + p[13]*t*t*v)
