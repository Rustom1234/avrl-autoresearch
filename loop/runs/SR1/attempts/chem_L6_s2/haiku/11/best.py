def equation(t, A, params):
    k1, K, c2, c1, d, q = params[0], params[1], params[2], params[3], params[4], params[5]
    return -k1*A/(K+A) + (c2*t**2 + c1*t)/A - d/A**q
