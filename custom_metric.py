

def Rsqured(x,y):
    import keras.backend as K

    x = K.batch_flatten(x)
    y = K.batch_flatten(y)

    avx = K.mean(x)
    avy = K.mean(y)

    num = K.sum((x-avx) * (y-avy))
    num = num * num

    denom = K.sum((x-avx)*(x-avx)) * K.sum((y-avy)*(y-avy))

    return num/denom

