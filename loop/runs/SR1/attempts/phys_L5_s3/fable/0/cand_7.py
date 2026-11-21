def equation(t, x, v, params):
    return params[0]-0.253*t**1.5+params[3]*x+params[4]*x**3+params[5]*np.abs(x)**params[6]+params[7]*np.abs(x)**params[8]