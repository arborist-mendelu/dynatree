"""
V pythonu vyres nasledujici:

* nacti sqlite databazi test.db, tabulku comments
* tabulka ma sloupce id, directory, image, text, rating, PRIMARY_KEY, 
* pro kazdy zapis, ktery ma v directory text utlum zkopiruj zaznam tak, aby byl direcotry roven utlum_vsechny_senzory a image se zmenil tak, ze pred koncovku .png se prida _Elasto(90). Vse ostatni zustane.
* uloz novou databazi jako test_new.db
"""

import sqlite3
import shutil
import os

# Záloha struktury databáze (bez dat)
source_db = 'test.db'
target_db = 'test_new.db'

# Zkopíruj původní databázi jako základ nové
shutil.copyfile(source_db, target_db)

# Připojení k nové databázi
conn = sqlite3.connect(target_db)
cursor = conn.cursor()

# Získání záznamů s directory = 'utlum'
cursor.execute("SELECT id, directory, image, text, rating FROM comments WHERE directory = 'utlum'")
rows = cursor.fetchall()

# Vložení upravených záznamů
for row in rows:
    id_, directory, image, text, rating = row
    # úprava directory a image
    new_directory = 'utlum_vsechny_senzory'
    if image.endswith('.png'):
        new_image = image[:-4] + '_Elasto(90).png'
    else:
        new_image = image  # pokud náhodou nemá .png

    # Vložení nového záznamu (bez PRIMARY_KEY – předpokládá se autoincrement)
    cursor.execute("""
        INSERT INTO comments (directory, image, text, rating)
        VALUES (?, ?, ?, ?)
    """, (new_directory, new_image, text, rating))

# Ulož změny a zavři připojení
conn.commit()
conn.close()
