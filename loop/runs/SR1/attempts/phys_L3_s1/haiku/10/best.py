def equation(t, x, v, params):
    C = np.cos(params[0]*t); S = np.sin(params[0]*t)
    return (params[1] + params[2]*v + params[3]*v**2 + params[4]*v**3
            + params[5]*C + params[6]*S + (params[7]*C + params[8]*S)*v + params[9]*x**3)
