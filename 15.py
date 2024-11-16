import numpy as np
import h5py 
import cv2


if __name__ == '__main__':

    parametros = h5py.File('theta_digitos.h5', 'r')

    imagen = cv2.imread('prueba03.jpg')

    # cv2.imshow('Prueba', imagen)
    # cv2.waitKey()

    # print(imagen.shape)
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    # print(imagen_gris.shape)\
    imagen_gris = cv2.GaussianBlur(imagen_gris, (5, 5), 0)
    ret, imagen_bn = cv2.threshold(imagen_gris, 90, 255, cv2.THRESH_BINARY_INV)

    # cv2.imshow('Prueba', imagen_bn)
    # cv2.waitKey()

    grupos, _ = cv2.findContours(imagen_bn.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(grupos)
    # print(len(grupos))
    ventanas = [cv2.boundingRect(g) for g in grupos]
    print(ventanas)
    
    for v in ventanas:
        cv2.rectangle(imagen, (v[0], v[1]), (v[0] + v[2], v[1] + v[3]), (255, 0, 0), 2)
        espacio = int(v[3] * 1.6)
        p1 = int((v[1] + v[3] // 2) - espacio // 2)
        p2 = int((v[0] + v[2] // 2) - espacio // 2)
        digito = imagen_bn[p1 : p1 + espacio, p2 : p2 + espacio]
        digito = cv2.resize(digito, (20, 20))
    
    cv2.imshow('Prueba', imagen)
    cv2.waitKey()