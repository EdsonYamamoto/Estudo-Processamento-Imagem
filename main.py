from aula import aula1
from aula import aula2
from aula import aula3
from aula import aula4
from aula import aula5
from aula import aula6


if __name__ == '__main__':
    texto='[1] aula\n[2] aula 2\n[3] aula 3\n[4] aula 4\n[5] aula 5\n[6] aula 6'
    print(texto)
    aula = input()
    if aula is '1':
        aula1.Aula.escolher(object)
    if aula is '2':
        aula2.Aula2.escolher(object)
    if aula is '3':
        aula3.Aula3.escolher(object)
    if aula is '4':
        aula4.Aula4.escolher(object)
    if aula is '5':
        aula5.Aula5.escolher(object)
    if aula is '6':
        aula6.Aula6.escolher(object)

