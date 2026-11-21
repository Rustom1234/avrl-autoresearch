def equation(t, A, params):
    return (params[0]+params[1]*t)*A + params[2]*A**2 + (params[3]+params[4]*t)/A + params[5] + params[6]*t