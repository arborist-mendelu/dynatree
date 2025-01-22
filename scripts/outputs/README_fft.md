# FFT spektrum


## Výstup skriptu `FFT_spectrum.py`

Skript kreslí pøedpoèítaná data z FFT rozkladu

* FFT_freq.csv - frekvence základních kmitù
* boxplot.pdf - frekvence základních kmitù, rozdìleno na olistìné a
  bez listí
* swarmplot.pdf - frekvence základních kmitù, rozdìleno na jednotlivé
  dny a jdou vidìt jednotlivá mìøení jako teèky


## Skripty pro rozklad pomocí FFT

* Skript `FFT_spectrum.py` funguje buï jako knihovna nebo jako skript,
  který analyzuje v¹echna mìøení. Ve druhém pøípadì se zaèátky a konce
  analyzovaného signálu nastavují podle informace v souboru
  `oscillation_times_remarks.csv` v podadresari `csv`.
* Skript `demo_FFT_spectrum.py` ukazuje, jak vykreslit jedno mìøení.
* Skript `dash_FFT.py` umo¾òuje zobrazit FFT analýzu ve www
  prohlí¾eèi, nastavit si rozsah, pou¾itý probe, zobrazit FFT i Welch
  spektrum.




