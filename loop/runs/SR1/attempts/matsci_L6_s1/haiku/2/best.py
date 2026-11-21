def equation(eps, T, params):
    E = np.abs(eps)
    return (params[0] + params[1]*np.log(T) + params[2]*E**3/T + params[3]/T
            + params[4]*E**3/T**2 + params[5]*E**3 + params[6]*E**2/T)
