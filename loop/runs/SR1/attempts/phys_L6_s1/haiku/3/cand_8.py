def equation(t, x, v, params):
    w=params[0]
    c=np.cos(w*t); s=np.sin(w*t)
    return params[1]*c + params[2]*s + params[3]*x + params[4]*v + params[5]*x**3 + (params[6]*c+params[7]*s)*v + params[8]*v**3 + params[9]
