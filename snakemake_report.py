import os
from rich.console import Console
from rich.table import Table
from datetime import datetime
import argparse

console = Console()


log_directory = '.snakemake/log/'  # Adresář s logy



# Nastavení argument parseru
parser = argparse.ArgumentParser(description="Analyzátor Snakemake logů.")
parser.add_argument('--logfile', type=str, help="Cesta k log souboru")
args = parser.parse_args()

# Použití zadaného souboru nebo výběr nejnovějšího
if args.logfile:
    latest_log = args.logfile
    if not os.path.exists(latest_log):
        console.print(f"[bold red]Soubor '{latest_log}' neexistuje.[/bold red]")
        exit()
else:
    log_files = [os.path.join(log_directory, f) for f in os.listdir(log_directory) if f.endswith('.log')]
    if not log_files:
        console.print("[bold red]Žádné logy nebyly nalezeny.[/bold red]")
        exit()
    latest_log = max(log_files, key=os.path.getmtime)

console.print(f"[bold green]Analyzuji soubor {latest_log}[/bold green]")


# # Najdi nejnovější log soubor
# log_files = [os.path.join(log_directory, f) for f in os.listdir(log_directory) if f.endswith('.log')]
# if not log_files:
#     print("Žádné logy nebyly nalezeny.")
#     exit()

# latest_log = max(log_files, key=os.path.getmtime)
# print(f"Analyzuji soubor: {latest_log}")

# Zpracování logu
durations = []
start_time = None
current_rule = None

with open(latest_log) as f:
    for line in f:
        # Detekce časové značky
        if line.startswith('[') and ']' in line:
            timestamp_str = line[1:line.index(']')]
            timestamp = datetime.strptime(timestamp_str, '%a %b %d %H:%M:%S %Y')
        
        # Detekce startu pravidla
        if "localrule" in line:
            current_rule = line.split()[1][:-1]  # Extrahuje název pravidla
            start_time = timestamp
        
        # Detekce konce pravidla
        elif "Finished job" in line and start_time and current_rule:
            duration = (timestamp - start_time).total_seconds()
            durations.append((current_rule, duration))
            start_time = None
            current_rule = None

durations.sort(key=lambda x: x[1], reverse=True)
            
# # Vytvoření tabulky pro výpis
table = Table(title="[red]Doba trvání Snakemake úloh[/red]", show_header=True, header_style="bold")
table.add_column("Pravidlo", style="cyan", no_wrap=True)
table.add_column("Doba trvání (s)", style="magenta", justify="right")

# Přidání řádků do tabulky
for rule, duration in durations:
    table.add_row(rule, f"{duration:.2f}")


console.print(table)


# Vyhledání nejdelší úlohy
if durations:
    longest = max(durations, key=lambda x: x[1])
    console.print(f'[bold yellow]Nejdelší úloha: {longest[0]}, Doba trvání: {longest[1]:.2f} sekund[/bold yellow]')

else:
    print('Nebyla nalezena žádná data o době běhu.')

