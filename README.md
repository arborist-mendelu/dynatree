# Zpracování dat

## Zdroje

Zdrojová data se přetransformují do parquet souborů. Skripty potom sahají jenom sem a do adresáře `skripty/csv`. Umístění parquet souborů vzheldem k adresáři skripty je
`../data`. Pravidelně se zrcadlí na jupyter.mendelu.cz do adresáře `/babice`, kde má práva pro čtení každý.

Skripty pro tvorbu parquet souborů:

* `acc2parquet.py`  acc na 100Hz, staci spustit
* `acc2parquet_full.py`  acc na 5000Hz, staci spustit
* `parquet_add_inclino.py` Prida tahovky k optice, resamplovani tahovek, casova synchronizace a nulovani inklinometru podle csv/synchronization_finetune_inclinometers_fix.csv, Staci spustit
* `find_tsv.py`, `xsight_tsv2parquet.py` Nalezeni a transformace tsv souboru. Staci spustit ve spravnem poradi (nejdriv najit adresare a potom je zpracovavat.)
* `pull2parquet.py`  Tahovky do parquet souboru. Nejdriv najit zdroje a pote spustit skript. Nalezeni zdroju viz navod v souboru.

Acc jsou nezavisle, prdavani tahovek k optice predpoklada, ze jsou aktualni data z tahovek a z optiky, tj. `parquet_add_inclino.py` spoustet nakonec. A vzdy po uprave 
`synchronization_finetune_inclinometers_fix.csv`.

## Skripty

Krome transformace zdroju (predchozi odstavec) je vsechno automatizovane. Staci spustit `snakemake` pripadne pravidlo `all`.

Mozna bude hlasit chybu ohledne temp adresaru. Obvykle se pouzivaji `./temp` a `../temp`, tak je dobre je vytvorit, pokud se to na nich zasekne. Vyzuivaji se nastroje pro 
praci s pdf souboru jako pdfunite apod. Doinstaluj, kde to hlasi chybu. Nebo zkontroluj spoustene prikazy podle snakefile.

## Solara aplikace

`solara run solara_app.py` - pripadne pomoci parametru nastavit jmeno serveru a port.

Pripadne jedna cast `solara run solara_FFT` apod. Vsechno ma zacatek "solara".

## Výstupy

Viz `snakefile` soubor. Výstupy jsou zpravidla v adresáři ../outputs. Jednou za čas se ručně přepkopíruje na erc disk do adresáře Mereni_Babice_zpracovani/vystupy (nebo tak nejak)



# Skripty pro zpracování dat z optiky apod - totó odnés čas ....

# Něco platí, něco ne.

* Soubory, které něco kreslí jsou soubory se jménem `plot_*.py`
kde se něco testuje a zkouší jsou soubory `temp_*.py` nebo
`untitled*.py`. Knihovny jsou soubory `lib_*.py`. Ostatní soubory jsou odladěné
soubory nebo knihovny pro zpracování dat.
* Některé soubory se dají použít jako knihovny a když se spustí jako programy,
  vyvíjí nějakou činnost, například upravují csv soubory z hlediska
  synchronizace. V takovém případě na toto na začátku běhu upozorní a pokračují
  až po pozitivní odpovědi. 

## Kopie csv souborů - platilo pro staré umístění, nově je potřeba přizpůsobit

Tahat data pokaždé přes síť je pomalé. Lepší je zkopírovat si data k sobě. 

Vytvořte si adresář, kam stáhnete data a kde budete pracovat. Poté běžte do
tohoto adresáře. 
Místo "babice" je možné si dát vlastní název. Může být v libovolném místě adresářové struktury.
```
mkdir babice
cd babice
mkdir skripty
mkdir data
```

Pokud máte QNAP disk s ERC daty přimountovaný, je možné použít rsync a stáhnout z
adresářů `Mereni_Babice_zpracovani/data` datové parquet soubory a textové soubory v
adresáři `pulling_tests`. Doba běhu podle rychlosti sítě, cca 5 minut na rychlém
připojení. 

```
rsync -zarv  -P --prune-empty-dirs --include "*/"  --include="*/*.TXT" --include="*/*.parquet" --exclude="*" /mnt/ERC/ERC/Mereni_Babice_zpracovani/data .
```
Všechno na jeden řádek, zdrojovou cestu `/mnt/ERC/ERC` upravit podle potřeby.
Uvedený příkaz platí, pokud je připojena následujícím příkazem (adresář `/mnt/ERC`
musí existovat).

```
sudo mount -t cifs //10.18.52.96/home /mnt/ERC/ -o ro,username=unod,uid=1000
```

Na QNAP serveru jsou skripty v adresáři `Mereni_Babice_zpracovani/skripty` ale na
názvu nezáleží. Je možno si stáhnout vše potřebné z GitHubu.

```
git clone git@github.com:robert-marik/dynatree-optika.git
```

Repozitář není veřejný, ale můžete tam mít přístup (email Robertovi).


## Krok 1.: Z Xsight do csv, `xsight_tsv2csv.py`, `find_tsv.py`

### `find_tsv.py`

Najde adresare s tsv soubory a zapise je spolu s datem a cislem stromu a mereni 
do databaze `csv/tsv_dirs.csv`

### `xsight_tsv2parquet.py`

Čte data uložená předchozím skriptem a ukládá meření do parquet souborů. To je 
sice binární fomrát, ale rychlejší a úspornější varianta csv.

### `xsight_tsv2csv.py`
UZ SE NEPOUZIVA.

Načtou se tsv soubory všechny soubory z jednoho měření se
převedou na jeden csv soubor. Skript `xsight_tsv2csv.py` by se měl
spustit vždy po přidání dat z optiky. Skript prochází adresáře s
domluvenými jmény a pro každé měření kontroluje existenci csv
soubor. Pokud csv soubor existuje, skript nic nedělá, jenom vypíše, že soubor
přeskakuje. Pokud csv soubor neexistuje, je vytvořen.

## Krok 2.: Přidání dat z inklinoměrů, `parquet_add_inclino.py`

Následující komentář je pro csv soubory. Od 16.8.2024 předěláno 
na parquet soubory, jinak vše zůstává v platnosti.

Doplnění dat z optiky se provádí přes soubory v adresáři
pulling_tests. Najde se maximální síla, maximální výchylka `Pt3`, tato
maxima se sesynchronizují, data se přepočítají tak, aby byla hodnota
ve stejných časových okamžicích jako jsou data z optiky a vše se uloží
do csv souboru do adresáře `csv_extended`. Kvůli úspoře místa a výkonu
se nespojuje s původním csv souborem z optiky. Synchronizaci je možno
vytunit zadáním explicitní opravy v souboru
`csv/synchronization_finetune_inclinometers_fix.csv`. Pokud nějaký
inklinometr poskočil, je možné zadat přes csv soubor interval, na
kterém má být střední hodnota inklinoměru nulová.

Pokud síla není naměřena, berou se pro synchronizaci začátky měření.

Kromě toho je možno dělat pohodlněji (zobrazit si graf ve vybrané velikosti, 
s vybraným rozsahem pro čas a s aktuálním zohledněním nastavení v souboru `csv/synchronization_finetune_inclinometers_fix.csv`)
pomocí programu `dash_force_inclino_sync.py` buď spuštěním v konzoli nebo ve Spyderu a následně na 
http://127.0.0.1:8050/ . 

Update: preferovanější varianta je `solara_force_elasto_inclino.py`.

Jedno měření je možno zobrazit příkazy jako například následující sada.

```
from plot_probes_inclino_force import plot_one_measurement

measurement = "M03"
tree = "BK08"
day = "2021-03-22"

plot_one_measurement(
        date=day,
        tree=tree, 
        measurement=measurement, 
        # xlim=(42,50),
        ) 
```

Při práci ve Spyderu  je možné nastavit zobrazování grafů v samostatném okně příkazem `%matplotlib qt` zadaným v konzoli
Spyderu. Potom je možné obrázek zvětšit a myší si zobrazit výřez. Zpět na původní 
funkci je možno příkazem `%matplotlib inline`

## Krok 3.: `FFT_spectrum.py`

Obsahuje proceduru pro FFT analýzu jednoho měření, proceduru pro
analýzu jednoho dne a zápis dat do souboru. Pokud je spuštěno jako
skript, dělá FFT analýzu pro všechna data a na konci spustí skript
`plot_fft.py` pro vytvoření celkového přehledu. V tomto přehledu se
nezohledňují základní frekvence, které jsou moc malé, nebo kde je moc
velký skok mezi sousedními frekvencemi (je krátký signál).

Kromě toho je možno dělat pohodlněji (zobrazit si graf ve vybrané velikosti, 
s vybraným rozsahem pro čas a s aktuálním zohledněním nastavení v souboru `csv/synchronization_finetune_inclinometers_fix.csv`)
pomocí programu `dash_FFT.py` buď spuštěním v konzoli nebo ve Spyderu a následně na 
http://127.0.0.1:8050/ .


## Release data

Skript `extract_release_data.py`


## Příkazy pro spojení obrázků do jednoho

Některé skripty si ukládají data na disk. Aby se daly obrázky spojit a aby byly
chronologicky, je výhodné je dát do adresářů číslovaných popořadě.
```
ln -s ../01_Mereni_Babice_22032021_optika_zpracovani 01
ln -s ../01_Mereni_Babice_29062021_optika_zpracovani 02
ln -s ../01_Mereni_Babice_05042022_optika_zpracovani 03
ln -s ../01_Mereni_Babice_16082022_optika_zpracovani 04
```

Spojení všech FFT diagramů pro strom na jednu stranu a převod na PDF je ve
skriptu `all_fft_to_one_page.sh`.


## Útlum

Knihovna `lib_damping.py` obsahuje funkce pro nalezení útlumu třemi
metodami:

* pomocí Hilbertovy transformace a vnější obálky
* pomocí proložení exponenciely body v maximech a minimech časového vývoje
kmitů
* pomocí waveletů

Pro útlum je nutné znát frekvenci kmitů a začátek a konec
analyzovaného signálu. . K tomuto se používají informace v csv
souborech `csv/oscillation_times_remarks.csv` a
`csv/results_fft.csv`. První z těchto souborů je možno editovat ručně,
druhý je výstupem skriptu `FFT_spectrum.py`. Oba csv soubory jsou pod
version control.

Pokud je konec analyzovaného časového intervalu roven -1, signál se
neanalyzuje.

Pokud se knihovna `lib_damping.py` spustí jako skript, analyzuje
všechna měření a výsledek zapíše do csv souboru
`damping_output/damping_results.csv`. Tato data čte skript
`plot_utlum.py`, který kreslí útlum pro jednotlivé stromy a odlišuje
jednotlivé dny nebo stavy s listím a bez listí.

Aplikace `streamlit_damping.py` umožňuje vybrat si den, strom a měření
a zobrazí všechny tři výstupy plus Hilbertovu-Huangovu transformaci.

## 

