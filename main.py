from aula import aula1
from aula import aula2

if __name__ == '__main__':
    texto='[1] aula\n[2] aula 2'
    print(texto)
    aula = input()
    if aula is '1':
        aula1.Aula.escolher(object)
    if aula is '2':
        aula2.Aula2.escolher(object)


