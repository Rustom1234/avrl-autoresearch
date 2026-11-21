def equation(t,P,params):
    a,b,c,d,e,f,g=params[0],params[1],params[2],params[3],params[4],params[5],params[6]
    return P*(a+b*np.exp(-c*t))-d*P**2+e*t+f+g*t**2