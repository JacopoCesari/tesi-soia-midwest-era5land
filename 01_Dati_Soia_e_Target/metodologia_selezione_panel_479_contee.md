# Nota metodologica - selezione del balanced panel county-level

## Obiettivo della selezione

L'obiettivo è costruire un dataset county-level adatto alla previsione della resa della soia con modelli di machine learning/deep learning, mantenendo simultaneamente: (i) una serie storica lunga, (ii) un numero elevato di contee, (iii) una copertura geografica concentrata in aree storicamente rilevanti per la coltivazione della soia e (iv) assenza di valori target mancanti nel campione principale.

La selezione non è stata effettuata sulla base della sola importanza produttiva corrente delle contee, perché ciò avrebbe favorito la geografia agricola contemporanea e avrebbe potuto introdurre un campione poco rappresentativo dell'intero orizzonte storico. Sono stati inizialmente considerati sei stati caratterizzati da una presenza storica importante e persistente della soia: Illinois, Indiana, Iowa, Minnesota, Missouri e Ohio.

## Pulizia dei dati USDA NASS

I dati county-level sono stati ottenuti da USDA NASS Quick Stats. Per il target sono state mantenute le osservazioni SURVEY relative a `SOYBEANS - YIELD, MEASURED IN BU / ACRE`. Le osservazioni espresse per `NET PLANTED ACRE` e le unità aggregate `OTHER COUNTIES` / `OTHER (COMBINED) COUNTIES` sono state escluse, poiché non corrispondono alla definizione di target scelta o non sono associabili univocamente a una geometria di contea. Per la superficie sono state mantenute le osservazioni di `SOYBEANS - ACRES HARVESTED`. Le osservazioni CENSUS sono conservate separatamente e non vengono fuse automaticamente con la serie SURVEY annuale.

Gli identificatori geografici sono stati normalizzati tramite county FIPS. È stata inoltre armonizzata la discontinuità storica dell'identificatore di Ste. Genevieve County, Missouri, ricondotta al FIPS 29186, in modo che la stessa unità geografica non fosse trattata come due contee distinte.

## Scelta del periodo e delle contee

È stato imposto un vincolo di balanced panel: una contea può appartenere al dataset principale soltanto se possiede contemporaneamente una stima SURVEY di yield e una stima SURVEY di acres harvested per ogni anno dell'intervallo considerato. Sono stati confrontati diversi possibili anni finali mantenendo il 1950 come anno iniziale, coerente con la disponibilità di ERA5-Land.

Il periodo 1950-2010 è stato scelto perché massimizza in modo favorevole la quantità di dati utilizzabili sotto il vincolo di panel completamente bilanciato. In questo intervallo rimangono 479 contee con 61 annualità complete, per un totale di 29.219 osservazioni county-year. Le stesse 479 contee possiedono sia yield sia acres harvested in ogni anno del periodo, senza necessità di imputare il target.

La scelta privilegia quindi la numerosità del dataset utile al training, aspetto particolarmente rilevante in vista dell'impiego di modelli di deep learning, evitando contemporaneamente un panel sbilanciato in cui la composizione geografica del campione cambia da un anno all'altro.

## Conservazione degli anni esclusi dal panel principale

Gli anni 2011-2025 non vengono eliminati dai file puliti. Tutte le osservazioni SURVEY disponibili per le 479 contee selezionate sono conservate in un foglio separato (`all_years_survey`). Tali anni non fanno parte del balanced panel principale perché alcune contee non dispongono di una stima NASS in tutti gli anni, ma vengono mantenuti per consentire future analisi di sensitività o una successiva modifica dell'orizzonte temporale senza dover ripetere il download e la pulizia dei dati.

Non viene effettuata alcuna imputazione dei valori NASS mancanti negli anni 2011-2025.

## Gestione dei dati meteorologici da adottare

La lista delle 479 contee deve essere considerata fissa per la costruzione del dataset meteorologico. I dati ERA5/ERA5-Land dovranno essere scaricati indipendentemente dal target NASS e conservati a un livello sufficientemente grezzo da permettere di cambiare successivamente aggregazione temporale o periodo di analisi.

La pipeline raccomandata è:

1. conservare un archivio meteorologico raw, preferibilmente orario, per tutte le celle ERA5-Land necessarie a coprire le 479 contee;
2. associare in modo stabile ciascuna cella ERA5-Land alle contee mediante FIPS e una matrice di pesi spaziali precomputata;
3. se disponibile, costruire pesi spaziali orientati alle aree tipicamente coltivate a soia, mantenendoli fissi nel tempo per non introdurre informazione futura;
4. derivare localmente le variabili meteorologiche non lineari prima dell'aggregazione spaziale quando necessario (ad esempio VPD, GDD, heat stress, ore sopra soglia, dry spells);
5. costruire un dataset giornaliero county-level come livello intermedio canonico e da questo derivare aggregazioni settimanali, cumulative e rolling;
6. produrre le feature ai diversi cutoff stagionali usando esclusivamente dati meteorologici disponibili fino al cutoff, evitando leakage temporale;
7. usare come dataset principale di modellazione l'intersezione `479 contee × 61 anni = 29.219 county-year` del periodo 1950-2010;
8. mantenere separati gli anni 2011-2025 per eventuali esperimenti successivi, senza modificare i raw meteorologici già scaricati.

Questa organizzazione separa la fase costosa di acquisizione meteorologica dalla scelta finale del periodo di modellazione: i raw ERA5-Land vengono scaricati una sola volta, mentre balanced panel, aggregazioni temporali e cutoff vengono generati localmente e possono essere modificati senza nuovi download.
