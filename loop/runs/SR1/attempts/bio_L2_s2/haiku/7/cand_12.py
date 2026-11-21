def equation(t,P,params):
    a,b,c,d,e,f,g=params[0],params[1],params[2],params[3],params[4],params[5],params[6]
    return P*(a+b*np.sin(c*t+d))-e*P**2+f+g*np.sin(c*t+d)