def equation(t, x, v, params):
    p = params
    a = np.abs(v)
    return (2.43*p[0]*np.sin(0.567*p[1]*t)*np.sqrt(a) - 0.133*p[2]*t**2 + 13.2*p[3]
            - 7.54*p[4]*a**(0.36*p[5]) + 0.62*p[6]*v + 0.01*(p[7]-1)*x + 0.01*(p[8]-1)*x**3 + 0.01*(p[9]-1)*x*v)