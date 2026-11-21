def equation(t, P, params):
    w=params[5]
    base=P*(params[0]+params[1]*P+params[2]*P**2)
    mod=P*(params[3]*np.sin(w*t)+params[4]*np.cos(w*t))
    return base+mod
