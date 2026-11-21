def equation(t, x, v, params):
    w=params[0]
    c=np.cos(w*t); s=np.sin(w*t)
    return params[1]*c + params[2]*s + params[3] + params[4]*x + params[5]*v + params[6]*x**3 + (params[7]*c+params[8]*s)*v + params[9]*x**2
