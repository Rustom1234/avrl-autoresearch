def equation(t, A, params):
    lt=np.log(t)
    return -params[0]*A**params[1]*np.exp(params[2]*lt+params[3]*lt**2+params[4]*lt*np.log(A))
