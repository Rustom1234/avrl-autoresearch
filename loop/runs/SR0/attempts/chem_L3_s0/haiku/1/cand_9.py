def equation(t, A, params):
    return (params[0]+params[1]*t)*A**params[2] + params[3]*A**params[4] + params[5]*A*(t)**0.5 + params[6]*A**2*t