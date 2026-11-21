def equation(t, A, params):
    return -params[0]*np.abs(A)**params[1]*np.abs(t)**params[2]