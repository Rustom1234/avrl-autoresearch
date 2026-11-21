def equation(t, P, params):
    r, K, A = params[0], params[1], params[2]
    return r*P*(1-P/K)*(P/A-1)