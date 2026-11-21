def equation(eps, T, params):
    A=np.abs(eps)
    return -(params[0]+params[1]*T+params[2]*T**2+params[3]*A+params[4]*A**2+params[5]*T*A+params[6]*T*A**2+params[7]*T**2*A+params[8]*T**2*A**2+params[9]*np.log(A))
