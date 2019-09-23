from tkinter import *
import tkinter as tk
import time
import random as rand

def desenha_planeta(x, y, raio, cor):
    canvas.create_oval(x-raio, y-raio, x+raio, y+raio, fill = cor)
    
def desenha_orbita(x, y, raio):
    canvas.create_oval(x-raio, y-raio, x+raio, y+raio, dash=(4, 2), outline="white")
    
def array_orbita(raio):
    orbita = []
    x = 0
    while x < largura:
        x_2 = (x-raio)*(x-raio)
        r_2 = raio*raio
        r_x_p = (pow(r_2-x_2,1/2))+altura/2
        r_x_n = ((-1)*pow(r_2-x_2,1/2))+altura/2
        if r_x_p.imag == 0 and r_x_n.imag == 0:
            temp = [int(x+largura/2)-raio,int(r_x_p),int(r_x_n)]
            if temp not in orbita:
                orbita.append(temp)
        x += 0.02
    return orbita

def move():
    canvas.delete("all")
    for i in planetas:
        if i[8] == 1:
            i[4] = i[6][i[7]][i[8]]
            i[5] = i[6][i[7]][0]
        else:
            i[4] = i[6][len(i[6])-i[7]-1][i[8]]
            i[5] = i[6][len(i[6])-i[7]-1][0]
        i[7] += 1
        if i[7]>=len(i[6]):
            i[7] = 0
            if i[8] == 1:
                i[8] = 2
            else:
                i[8] = 1
        desenha_orbita(altura/2, largura/2, i[1])
        desenha_planeta(altura/2, largura/2, 50, "yellow")
        desenha_planeta(i[4], i[5], i[0], i[3])
    tela.after(15, move)  
    
tela = Tk()
altura = 900
largura = 900

planetas = []

canvas = tk.Canvas(tela, width=largura, height=altura, borderwidth=0, highlightthickness=0, bg="black")
canvas.grid()

desenha_planeta(altura/2, largura/2, 50, "yellow")

color_list = ["white","red","blue","orange","purple","green","pink"]

for i in range(10):
    tamanho = rand.randint(i+2,i+10)
    if i != 0:
        orbita = rand.randint(planetas[i-1][1]+planetas[i-1][0]+tamanho,
                              planetas[i-1][1]+planetas[i-1][0]+tamanho+planetas[i-1][2])
    else:
        orbita = rand.randint(60,65)
    cor = color_list[rand.randint(0,6)]
    cons = rand.randint(30,45)
    temp = [tamanho, orbita, cons, cor, (largura/2)+orbita, altura/2, array_orbita(orbita), 0, 1,]
    planetas.append(temp)

tela.after(1, move)
tela.mainloop()