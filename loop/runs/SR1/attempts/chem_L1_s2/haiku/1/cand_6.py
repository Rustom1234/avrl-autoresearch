def equation(t, A, params):
    return -params[0]*(A*t)**params[1]/(1+params[2]*A*t)