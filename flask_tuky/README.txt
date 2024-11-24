Spoustej pomoci "flask run"

Ocekava soubor images.csv, kde jsou v jednom sloupci obrazky pomoci filename
a ve druhem sloupci valid jako True/False. Pokud je sloupcu vice, zahodi se.
Nastaveni jak ze jmena souboru udelat obrazek s celou adresou je v app.py.
Koncovka se predpoklada png.

Soubor images.csv se pri praci meni, tj. pokud se jedna o seriozni praci,
pracuj na kopii dat. Mimo jine se odstrani sloupce, ktere nejsou filename
nebo valid.