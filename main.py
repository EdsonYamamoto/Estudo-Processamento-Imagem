from aula import aula1
from aula import aula2
from aula import aula3

if __name__ == '__main__':
    texto='[1] aula\n[2] aula 2\n[3] aula 3'
    print(texto)
    aula = input()
    if aula is '1':
        aula1.Aula.escolher(object)
    if aula is '2':
        aula2.Aula2.escolher(object)

    if aula is '3':
        aula3.Aula3.escolher(object)

