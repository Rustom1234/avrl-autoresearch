def equation(t, P, params):
    L = np.log(t + params[4])
    return params[0] + params[1]*P + params[2]*P**2 + params[3]*L + params[5]*P*L + params[6]*P**2*L + params[7]*t
