def equation(t, P, params):
    return P*(params[0]+params[1]*np.sin(params[2]*t+params[3])) + params[4]*P*P + params[5]*P**3 + params[6] + params[7]*t + params[8]*t*P*P + params[9]*t*P