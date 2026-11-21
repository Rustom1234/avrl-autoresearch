def equation(t, x, v, params):
    w=params[0]
    c=np.cos(w*t); s=np.sin(w*t)
    return (params[1] + params[2]*x + params[3]*v + params[4]*x**2 + params[5]*x*v + params[6]*v**2
            + (params[7] + params[8]*x + params[9]*v)*c + (params[7]*0 + params[1]*0 + 0)*s)