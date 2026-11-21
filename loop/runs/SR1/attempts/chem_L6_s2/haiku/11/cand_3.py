def equation(t, A, params):
    k1, n, c, m, p = params[0], params[1], params[2], params[3], params[4]
    return -k1*A**n + c*t**m/A**p
