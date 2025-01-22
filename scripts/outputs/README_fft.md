# FFT spektrum


## V�stup skriptu `FFT_spectrum.py`

Skript kresl� p�edpo��tan� data z FFT rozkladu

* FFT_freq.csv - frekvence z�kladn�ch kmit�
* boxplot.pdf - frekvence z�kladn�ch kmit�, rozd�leno na olist�n� a
  bez list�
* swarmplot.pdf - frekvence z�kladn�ch kmit�, rozd�leno na jednotliv�
  dny a jdou vid�t jednotliv� m��en� jako te�ky


## Skripty pro rozklad pomoc� FFT

* Skript `FFT_spectrum.py` funguje bu� jako knihovna nebo jako skript,
  kter� analyzuje v�echna m��en�. Ve druh�m p��pad� se za��tky a konce
  analyzovan�ho sign�lu nastavuj� podle informace v souboru
  `oscillation_times_remarks.csv` v podadresari `csv`.
* Skript `demo_FFT_spectrum.py` ukazuje, jak vykreslit jedno m��en�.
* Skript `dash_FFT.py` umo��uje zobrazit FFT anal�zu ve www
  prohl�e�i, nastavit si rozsah, pou�it� probe, zobrazit FFT i Welch
  spektrum.




