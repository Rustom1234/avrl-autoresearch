def equation(t, A, params):
    return params[0]*np.abs(A)**params[1]/(params[2]+np.abs(A)**params[1]) + params[3]*t**2 + params[4]*t**3/A