# PF4EA generator & solver

Il codice di questa repository permette di generare e risolvere un'istanza del problema PF4EA
(PathFinding for an Entry Agent).

## Esecuzione singola

Il programma è stato scritto in Python (3) e può essere invocato da CLI assieme ai suoi argomenti.
Segue un esempio di chiamata con il significato degli argomenti nelle loro opportune posizioni:

```
$ python3 main.py <width> <height> <num_agents> \
                  <obstacle_ratio> <conglomeration_ratio> \
                  <agent_path_length> <entry_agent_max_length> \
                  <agent_generator> <heuristic> <seed>
```
* `width` e `height`: dimensioni della griglia.
* `num_agents`: numero di agenti già presenti.
* `obstacle_ratio`: percentuale di ostacoli espressa come numero tra 0 e 1.
* `conglomeration_ratio`: numero tra 0 e 1 che esprime quanto gli ostacoli siano raggruppati tra loro
lungo le direzioni cardinali.
* `agent_path_length`: massima lunghezza del percorso degli agenti già presenti.
* `entry_agent_max_length`: massima lunghezza della soluzione (il cammino dell'entry agent).
* `agent_generator`: strategia di generazione degli agenti. Una tra "random" e "optimal".
* `heuristic`: euristica adottata dal risolutore. Una tra "diagonal" e "dijkstra".
* `seed`: numero usato per inizializzare il generatore di numeri pseudo-casuali.

Ogni chiamata stampa a schermo su `stdout` i risultati dell'esecuzione:
* Stato (successo, fallimento, timeout)
* Lunghezza della soluzione
* Costo della soluzione
* Numero di stati espansi dall'algoritmo risolutivo
* Numero di stati inseriti dall'algoritmo risolutivo
* Mosse di attesa
* Tempo impiegato per generare la griglia
* Tempo impiegato per generare gli agenti
* Tempo impiegato per risolvere l'istanza

Il timeout predefinito è di 1 secondo (può essere modificato in `main.py`). In caso di timeout
il valore restituito da tutti gli altri attributi è forzato a 0 per evitare inconsistenze.

## Esecuzione in batch

Lo script bash `asd-grid.sh` può essere usato per sottoporre automaticamente al programma i parametri
di un gran numero di istanze da generare e risolvere, per poi raccogliere tutti i dati in un comodo .csv
dal nome a piacere sfruttando la redirezione dell'output.

```
$ ./asd-grid.sh > 42.1.csv
```

Si tenga presente che a fronte di un timeout predefinito di 1 secondo e dei valori usati nello script bash,
l'esecuzione di `asd-grid.sh` dovrebbe essere in grado di raccogliere i risultati di almeno 200000 istanze.
Un esempio di tale file è presente in `42.1.csv` (42 è il seed e 1 il timeout).

I parametri di timeout e sperimentazione possono essere facilmente modificati a mano in `main.py` e `asd-grid.sh`.