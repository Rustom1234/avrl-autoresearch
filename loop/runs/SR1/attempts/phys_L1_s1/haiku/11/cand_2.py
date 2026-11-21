
def equation(t, x, v, params):
    return params[0]*x**3 + params[1]*v*np.abs(v)
