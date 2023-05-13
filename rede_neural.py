""" 
    Laboratório de Inteligencia Artificial - Exercicio de redes neurais
    Professor Rogerio
    Alunos: Samuel Correa
            Caio Loot
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
import random

class redeNeural():
    def __init__(self, NOMEARQUIVO: str, col: int):
        linha_atual = 1
        colunas = col
        self.Iris_setosa = []
        self.Iris_versicolor = []
        self.Iris_virginica = []

        with open(NOMEARQUIVO, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for row in reader:
                linha = []
                if(linha_atual == 1):
                    linha_atual = linha_atual +1
                    pass
                elif(linha_atual <=51):
                    linha_atual = linha_atual +1
                    for i in range(colunas):
                        linha.append(float(row[i]))
                    self.Iris_setosa.append(linha)
                elif(linha_atual <=101):
                    linha_atual = linha_atual +1
                    for i in range(colunas):
                        linha.append(float(row[i]))
                    self.Iris_versicolor.append(linha)
                elif(linha_atual <=151):
                    linha_atual = linha_atual +1
                    for i in range(colunas):
                        linha.append(float(row[i]))
                    self.Iris_virginica.append(linha)
        
        self.dic_saida = {'Iris_setosa': [1, 0, 0],'Iris_versicolor': [0, 1, 0],'Iris_virginica': [0, 0, 1]}
        self.treinamento = []
        self.validador = []
        self.label_validador = []

        for i in range(len(self.Iris_setosa)):
            if(i <= 35):
                self.Iris_setosa[i].append([1, 0, 0])
                self.treinamento.append(self.Iris_setosa[i])
            else:
                self.validador.append(self.Iris_setosa[i])
                self.label_validador.append([1, 0, 0])
        for i in range(len(self.Iris_versicolor)):
            if(i <= 35):
                self.Iris_versicolor[i].append([0, 1, 0])
                self.treinamento.append(self.Iris_versicolor[i])
            else:
                self.validador.append(self.Iris_versicolor[i])
                self.label_validador.append([0, 1, 0])
        for i in range(len(self.Iris_virginica)):
            if(i <= 35):
                self.Iris_virginica[i].append([0, 0, 1])
                self.treinamento.append(self.Iris_virginica[i])
            else:
                self.validador.append(self.Iris_virginica[i])
                self.label_validador.append([0, 0, 1])
                
        np.random.shuffle(self.treinamento)
        self.label_treinamento = [linha[-1] for linha in self.treinamento]
        for linha in self.treinamento:
            linha.pop()
            
        self.erro_era = []
        self.erro_teste = []
        
        self.W = [[random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)], 
                  [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)], 
                  [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]]
        self.b = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]
        self.Y = [0, 0, 0]
        self.taxa_atualização = 0.01
        self.perceptron = perceptron()

    def teste(self):
        erro_do_teste = 0
        for i in range(len(self.validador)):
            self.Y[0] = self.perceptron.sigmoidal(pesos = self.W[0], entrada = self.validador[i], bias = self.b[0])
            self.Y[1] = self.perceptron.sigmoidal(self.W[1], self.validador[i], self.b[1])
            self.Y[2] = self.perceptron.sigmoidal(self.W[2], self.validador[i], self.b[2])
            resposta = self.calculo_resposta()
            if not resposta == self.label_validador[i]:
                erro_do_teste = erro_do_teste + 1
        self.erro_teste.append(erro_do_teste)
    
    def era(self):
        erro_da_era = 0
        for i in range(len(self.treinamento)):
            self.Y[0] = self.perceptron.sigmoidal(pesos = self.W[0], entrada = self.treinamento[i], bias = self.b[0])
            self.Y[1] = self.perceptron.sigmoidal(pesos = self.W[1], entrada = self.treinamento[i], bias = self.b[1])
            self.Y[2] = self.perceptron.sigmoidal(pesos = self.W[2], entrada = self.treinamento[i], bias = self.b[2])
            label = self.label_treinamento[i]
            erro = [label[0]-self.Y[0], label[1]-self.Y[1], label[2]-self.Y[2]]
            resposta = self.calculo_resposta()
            if not resposta == self.label_treinamento[i]:
                erro_da_era = erro_da_era + 1
            self.novo_w(erro, self.treinamento[i])
            self.novo_bias(erro)
        self.erro_era.append(erro_da_era)
        self.teste()

    def calculo_resposta(self):
        maior = 0
        for i in range(len(self.Y)):
            if self.Y[maior] < self.Y[i]:
                maior = i
        resultado = []
        for i in range(len(self.Y)):
            resultado.append(0)
        resultado[maior] = 1
        return resultado
    def novo_w(self, erro, X):
        #w novo = w velho + taxa att * erro * X
        self.W = np.array(self.W) + self.taxa_atualização * np.array([erro]).T * np.array(X)

    def novo_bias(self, erro):
        #bias novo = bias velho + taxa att * erro * X
        self.b = np.array(self.b) + self.taxa_atualização * np.array(erro)

class perceptron():
    def __init__(self):
        pass
    
    def somatorio(self, pesos, entrada, bias):
        saida = 0
        for i in range(len(entrada)):
            saida = saida + entrada[i]*pesos[i]
        return saida + bias

    def degrau(self, pesos, entrada, bias):
        valor = self.somatorio(pesos, entrada, bias)
        if(valor >= 0):
            return 1
        else:
            return 0
    def sigmoidal(self, pesos, entrada, bias):
        valor = self.somatorio(pesos, entrada, bias)
        return  1 / (1 + np.exp(-valor))

ITERACOES = 1000
NOMEARQUIVO = 'Iris_Data.csv'
rede = redeNeural('Iris_Data.csv', 4)
for i in range(ITERACOES):
    rede.era()
geração = []
for i in range(len(rede.erro_era)):
    geração.append(i)
print('erros calculados:')
print(rede.erro_era)
plt.plot(geração, rede.erro_era)
plt.show()
print(rede.erro_teste)
plt.plot(geração, rede.erro_teste)
plt.show()
print('rede treinada: ')
print('W')
print(rede.W)
print('bias')
print(rede.b)