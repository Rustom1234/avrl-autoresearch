def equation(eps, T, params):
    return -(params[0]*eps + params[1]*eps**2 + params[2]*T*eps + params[3]*T*eps**2 + params[4]*T**2*eps)
