def equation(t, A, params):
    k1, K, b, c, d, n = params[0], params[1], params[2], params[3], params[4], params[5]
    return -k1*np.exp(-b*t)*A/(1+K*A) + c*np.exp(-d*t)/A**n
