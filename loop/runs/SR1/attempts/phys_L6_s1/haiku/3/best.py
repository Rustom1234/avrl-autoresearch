def equation(t, x, v, params):
    w=params[0]
    c=np.cos(w*t); s=np.sin(w*t); c2=np.cos(2*w*t); s2=np.sin(2*w*t)
    return (params[1]*c + params[2]*s + params[3] + params[4]*x + params[5]*v + params[6]*x**3
            + params[7]*c2 + params[8]*s2 + (params[9]*c + params[7]*0 + 0*s)*v)
