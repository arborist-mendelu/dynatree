%% Nastaveni
prefix = "/mnt/ERC/ERC/Mereni_Babice_zpracovani/data";
day = "2021-03-22";
tree = "BK01";
measurement = "M03";
measurement_type = "normal";  % Varianty jsou normal, den, noc, afterro, afterro2, mokro, mraz

%% Načtení optiky
% např. /mnt/ERC/ERC/Mereni_Babice_zpracovani/data/parquet/2021_03_22/BK08_M04.parquet
% Zajima nas Pt3 a Pt4, pohyb v Y0. Ostatni nacitat nebudeme, dlouho by to
% trvalo. Matlab lehce zmrsi nazvy sloupcu, ale to si muzes potom
% prejmenovat.
filename = prefix+"/parquet/"+strrep(day, '-', '_')+"/"+tree+"_"+measurement+".parquet";
SelVarNames = ["('Pt3', 'Y0')","('Pt4', 'Y0')"];
T = parquetread(filename,'SelectedVariableNames',SelVarNames);


%% Načtení tahovek
% např. /mnt/ERC/ERC/Mereni_Babice_zpracovani/data/parquet_pulling/2021_03_22/normal_BK04_M03.parquet
% Zajima nas Force, Elasto, a z Inclino80 a Inclino81 vzdy jenom jeden, ale
% pokazdy jiny. Vyexportuju, ktery to je, zatim inklinometry preskoc.

filename = prefix+"/parquet_pulling/"+strrep(day, '-', '_')+"/"+measurement_type+"_"+tree+"_"+measurement+".parquet";
T = parquetread(filename);

%% Načtení ACC, 5000Hz (original)
% To josu vlastne ty data, ktera jsou v mat souborech, ale bylo by lepsi pro
% umisteni dat do repozitare pracovat jenom s temi parquet.
% /mnt/ERC/ERC/Mereni_Babice_zpracovani/data/parquet_acc/den_2022-08-16_BK08_M03_5000.parquet;
filename = prefix+"/parquet_acc/"+measurement_type+"_"+day+"_"+tree+"_"+measurement+"_5000.parquet";
T = parquetread(filename);






