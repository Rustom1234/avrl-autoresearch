def equation(t, P, params):
    H, k1, a, b, k2, K = params[0], params[1], params[2], params[3], params[4], params[5]
    g1 = 1-np.exp(-k1*t)
    g2 = 1-np.exp(-k2*t)
    return -H*g1 + (a+b*g2)*P*(1-P/K)
