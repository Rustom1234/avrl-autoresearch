def equation(t,P,params):
    return params[0]*P*(1-P/params[1]) - params[2]*np.tanh(t/params[3])*(1+params[4]*np.exp(-P/params[5]))*params[6]/(1+0*P) - params[7]*P*np.exp(-t/params[8])