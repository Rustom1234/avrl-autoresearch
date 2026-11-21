def equation(eps, T, params):
    e=eps; t=T
    return -(params[0]*e + params[1]*e**2 + params[2]*e*(t-params[3])**2 + params[4]*e**2*t + params[5]*e**3 + params[6]*e*t)