def equation(t, P, params):
    Q = np.abs(P)**params[6]
    return params[0] + params[1]*Q + params[2]*t + params[3]*t**2 + params[4]*t**3 + params[5]*Q*t + params[7]*Q*t**2
