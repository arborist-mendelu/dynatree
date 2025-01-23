Skript spouštět z nadřazeného adresáře a nastavit cesty .

```
export PYTHONPATH=/home/marik/dynatree/scripts:$PYTHONPATH
```
nebo 
```
export PYTHONPATH=`pwd`:$PYTHONPATH
```
a spoštět s cestou. 
```
python Lukes/Lukes_static_pull_get_Fmax.py
```