def f1(a, b):
    value = 0
    if a > 289349:
        if b/4 + 4 == 5024:
            raise Exception("Fatal Error")
        elif b/5 == 5024:
            value = 0
        else:
            value = 0
    else:
        value = 0
    return value

def f2(a, b):
    value = 0
    if a > 289349:
        if b < 1000230:
            raise Exception("Fatal Error")
        else:
            value = 0
    else:
        value = 0
    return value

def f3(a, b):
    value = 0
    if a == 289349:
        if b == 1000230:
            raise Exception("Fatal Error")
        else:
            value = 0
    else:
        value = 0
    return value

def f4(a, b, c):
    value = 0
    if a == 39923:
        if b == 112923:
            if c == 3:
                raise Exception("Fatal Error")
            else:
                value = 0
        else:
            value = 0
    else:
        value = 0
    return value

def f5(a, b, c):
    value = 0
    if a > 39923:
        if b > 112923:
            if c > 3:
                raise Exception("Fatal Error")
            else:
                value = 0
        else:
            value = 0
    else:
        value = 0
    return value

def f6(a):
    if 133445/a == 3:
        return 1
    else:
        return 0

def f7(a):
    x = [1, 2, 3]
    return x[a]

def f8(a):
    return 394282/(a - 392)

def g1(a):
    return 193923.02939 + 2.0/float(a)

def g2(a: float, b: float):
    if a > 2.1023:
        if b < 20302.0:
            raise Exception("Fatal Error") 
        else:
            return 0.0
    else:
        return 0.0
