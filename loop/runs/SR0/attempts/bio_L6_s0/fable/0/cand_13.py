def equation(t, P, params):
    h=1.67568743*np.abs(params[5]); p=1.24796811*params[6]; m=1.89416588*params[7]
    return 0.166*params[0]*t**1.5+0.0157*params[1]*np.sqrt(t)*P**3-0.04171455*params[2]*P-0.00677953*params[3]*P**2+3.39799288*params[4]*P**m/(h+P)**p