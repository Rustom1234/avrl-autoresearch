def equation(t, A, params):
    k1, n1, b, c, m, p, d, q = params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7]
    return -k1*np.exp(-b*t)*A**n1 + c*t**m/A**p - d/A**q
