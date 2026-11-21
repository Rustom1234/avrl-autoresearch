def equation(t, x, v, params):
    return (params[0]+params[1]*np.sqrt(t)+params[2]*t+params[3]*x+params[4]*x**2+params[5]*x**3
            +params[6]*v+params[7]*v*np.abs(v)+params[8]*v*x+params[9]*v**2)