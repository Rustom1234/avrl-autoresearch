def equation(t, P, params):
    r, K, H, k, n = params[0], params[1], params[2], params[3], params[4]
    return r*P*(1-P/K) - H*t**n/(k**n+t**n)
