Skripty s optikou. Prilezitostne zazipovat aktualni verzi a vlozit do Teams.

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

