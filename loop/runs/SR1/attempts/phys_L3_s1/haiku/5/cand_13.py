def equation(t, x, v, params):
    w=params[1]
    c_=np.cos(w*t); s_=np.sin(w*t)
    return (params[0]*c_+params[2]*s_)*(1+params[7]*v) - params[3]*v - params[4]*v*np.abs(v) - params[5]*x - params[6]*x**3 + params[8]