Skripty jsou pro zpracování dat z optiky.

Soubory, které něco kreslí jsou soubory se jménem plot_*.py. Soubory,
kde se něco testuje a zkouší jsou soubory temp_*.py nebo
untitled*.py. Ostatní soubory jsou odladěné soubory nebo knihovny pro
zpracování dat.

## Krok 1.: Z Xsight do csv, `xsight_tsv2csv.py`

Načtou se tsv soubory všechny soubory z jednoho měření se
převedou na jeden csv soubor. Skript xsight_tsv2csv.py by se měl
spusti vždy po přidání dat z optiky. Skript prochází adresáře s
domluvenými jmény a pr každé měření kontroluje existenci csv
soubor. Pokud csv soubor existuje, skript nic nedělá. Pokud
neexistuje, csv soubor je vytvořen.

## Krok 2.: Přidání dat z inklinoměrů, `csv_add_inclino.py`

Doplnění dat z optiky se provádí přes soubory v adresáři
pulling_tests. Najde se maximální síla, maximální výchylka Pt3, tato
maxima se sesynchronizují, data se přepočítají tak, aby byla hodnota
ve stejných časových okamžicích jako jsou data z optiky a vše se uloží
do csv souboru do adresáře csv_extended. Kvůli úspoře místa a výkonu
se nespojuje s původním csv souborem z optiky. Synchronizaci je možno
vytunit zadáním explicitní opravy v souboru
csv/synchronization_finetune_inclinometers_fix.csv. Pokud nějaký
inklinometr poskočil, je možné zadat přes csv soubor interval, na
kterém má být střední hodnota inklinoměru nulová.

Pokud síla není naměřena, berou se pro synchronizaci začátky měření.

Kromě toho je možno dělat pohodlněji (zobrazit si graf ve vybrané velikosti, 
s vybraným rozsahem pro čas a s aktuálním zohledněním nastavení v souboru csv/synchronization_finetune_inclinometers_fix.csv)
pomocí programu dash_force_inclino_sync.py buď spuštěním v konzoli nebo ve Spyderu a následně na 
http://127.0.0.1:8050/ .

Jedno měření je možno zpracovat příkazy jako napříkald následující sada.
```
from plot_probes_inclino_force import plot_one_measurement

measurement = "M03"
tree = "BK08"
day = "2021-03-22"

plot_one_measurement(
        measurement_day=day,
        tree=tree, 
        tree_measurement=measurement, 
        # xlim=(42,50),
        ) 
```

## Krok 3.: `FFT_spectrum.py`

Obsahuje proceduru pro FFT analýzu jednoho měření, proceduru pro
analýzu jednoho dne a zápis dat do souboru. Pokud je spuštěno jako
skript, dělá FFT analýzu pro všechna data a na konci spustí skript
`plot_fft.py` pro vytvoření celkového přehledu. V tomto přehledu se
nezohledňují základní frekvence, které jsou moc malé, nebo kde je moc
velký skok mezi sousedními frekvencemi (je krátký signál).

Kromě toho je možno dělat pohodlněji (zobrazit si graf ve vybrané velikosti, 
s vybraným rozsahem pro čas a s aktuálním zohledněním nastavení v souboru csv/synchronization_finetune_inclinometers_fix.csv)
pomocí programu dash_force_inclino_sync.py buď spuštěním v konzoli nebo ve Spyderu a následně na 
http://127.0.0.1:8050/ .



## Příkazy pro spojení obrázků do jednoho


```
stromy=`ls */*fft/BK*png | grep "BK.*_" -o | sort | uniq`; for i in $stromy; do montage */*fft/$i*png -tile 3x4 -geometry +0+0 ${i}fft_all.png; done
```
```
stromy=`ls 01*zpracovani/*fft/BK*png | grep "BK.*_" -o | sort | uniq`; for i in $stromy; do montage 01*zpracovani/*fft/$i*png -tile 4x4 -geometry +0+0 ${i}fft_all.png; done
```
Obrazky do PDF souboru, osm obrazku na stranu.
```
montage 01_Mereni_Babice_16082022_optika_zpracovani/png_with_inclino/BK*png -tile 2x4 -geometry +0+0   D.pdf
```

Adresare cislovane podle chronologie.
```
ln -s ../01_Mereni_Babice_22032021_optika_zpracovani 01
ln -s ../01_Mereni_Babice_29062021_optika_zpracovani 02
ln -s ../01_Mereni_Babice_05042022_optika_zpracovani 03
ln -s ../01_Mereni_Babice_16082022_optika_zpracovani 04
```

Vsechny png soubory s fft analyzou pro jeden strom na jednu stranu PDF.
```
stromy=`ls 0*/*fft/BK*png | grep "BK.*_" -o | sort | uniq`; for i in $stromy; do montage 0*/*fft/$i*png -tile 4x4 -geometry +0+0 ${i}fft_all.png; done
convert *all.png fft.pdf

```

