def equation(t,P,params):
    a,b,c,d,e,f,g,h,k=params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7],params[8]
    return a*P+b*P**2+c*P**3+d+e*t+f*t**2+g*P*t+h*P*t**2+k*P**2*t