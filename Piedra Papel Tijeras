from random import choice
from bokeh.plotting import figure,show
from bokeh.io import push_notebook, show, output_notebook
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(verbose=False, warm_start=True)
opciones = ['piedra','papel','tijeras']
empate, jugador1, jugador2 = 0,0,0

def Winer(p1,p2):
    if p1 == p2:
        result = 0
    elif p1 == 'piedra' and p2 == 'tijeras':
        result = 1
    elif p1 == 'piedra' and p2 == 'papel':
        result = 2
    elif p1 == 'tijeras' and p2 == 'piedra':
        result = 2
    elif p1 == 'tijeras' and p2 == 'papel':
        result = 1
    elif p1 == 'papel' and p2 == 'piedra':
        result = 1
    elif p1 == 'papel' and p2 == 'tijeras':
        result = 2
    return result

def get_choice():
    return choice(opciones)

def unjuego():
    for i in range(11):
        player1 = get_choice()
        player2 = get_choice()
        if Winer(player1,player2) == 0:
            empate += 1
        elif Winer(player1,player2) == 1:
            jugador1 += 1
        elif Winer(player1,player2) == 2:
            jugador2 += 1
        print 'player1: %s/ player2: %s--- Winer: %s' % (player1, player2, Winer(player1,player2))

def resltados():
    print 'empates: ' + str(empate)
    print 'jugador1: ' + str(jugador1)
    print 'jugador2: ' + str(jugador2)
#resltados()

def str_to_bin(opcion):
    if opcion=='piedra':
        res=[1,0,0]
    elif opcion=='papel':
        res = [0,1,0]
    elif opcion=='tijeras':
        res = [0,0,1]
    return res

dat_x = list(map(str_to_bin,['piedra','tijeras','papel']))
dat_y = list(map(str_to_bin,['papel','piedra','tijeras']))

model = clf.fit([dat_x[0]], [dat_y[0]])
#print model

def play_and_learn(iters=10, debug=False):
    score = {'Win':0,'loose':0}
    dat_x = []
    dat_y = []

    for i in range(iters):
        player1 = get_choice()
        predict = model.predict_proba([str_to_bin(player1)])[0]

        if predict[0] >= 0.95:
            player2 = opciones[0]
        elif predict[1] >= 0.95:
            player2 = opciones[1]
        elif predict[2] >= 0.95:
            player2 = opciones[2]
        else:
            player2 = get_choice()

        if debug==True:
            print 'jugador1: %s/Modelo: %s --> %s' % (player1,predict,player2)

        winer = Winer(player1,player2)
        if debug==True:
            print 'comprobamos: PL1 VS PL2 %s' % winer

        if winer == 2:
            dat_x.append(str_to_bin(player1))
            dat_y.append(str_to_bin(player2))
            score['Win']+=1
        else:
            score['loose']+=1
    return score,dat_x,dat_y


'''print dat_x
print dat_y
print 'score: %s %s %%' % (score, (score['Win']*100/(score['Win']+score['loose'])))
if len(dat_x):
    Model = model.partial_fit(dat_x,dat_y)'''

i = 0
historic_pct = []
while True:
    i+=1
    score, dat_x, dat_y = play_and_learn(1000, debug=False)
    pct = (score['Win']*100/(score['Win']+score['loose']))
    historic_pct.append(pct)
    print 'Iter: %s - Score: %s %s %%' % (i, score, pct)

    if len(dat_x):
        Model = model.partial_fit(dat_x,dat_y)

    if sum(historic_pct[-9:])==900:
        break

x = range(len(historic_pct))
y = historic_pct

p = figure(
    title='Porcentaje de aprendizaje',
    x_axis_label='Iter', y_axis_label='%', width=900)
p.line(x,y,legend=None, line_width=1)
show(p)
