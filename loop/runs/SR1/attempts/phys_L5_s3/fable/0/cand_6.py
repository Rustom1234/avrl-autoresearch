def equation(t, x, v, params):
    return params[0]-0.253*t**1.5+params[3]*x+params[4]*x**3+params[5]*np.sin(params[6]*x+params[7])