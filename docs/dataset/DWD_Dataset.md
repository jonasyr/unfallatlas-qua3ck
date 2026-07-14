# DATASET DESCRIPTION

**Hourly station observations of weather phenomena for Germany**

- **Version:** v24.03
- **Publication date:** 2024-03-29
- **Cite data set as:** Hourly station observations of weather phenomena for Germany, Version v24.03
- **Dataset-ID:** urn:x-wmo:md:de.dwd.cdc::obsgermany-climate-hourly-weather_phenomena

Dataset URLs:
- https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/weather_phenomena/historical/
- https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/weather_phenomena/recent/
- https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/weather_phenomena/recent/WW_Stundenwerte_Beschreibung_Stationen.txt
- https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/weather_phenomena/recent/Wetter_Beschreibung.txt
- https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/weather_phenomena/timeseries_overview

---

## Abstract

These data originate from the stations of the DWD and legally as well as qualitatively equal partner network stations. Extensive station metadata, such as station relocations, instrument changes, reference time changes, algorithm changes or operator information are included.

The dataset is divided into a versioned part with completed quality check, in the directory `./historical/`.
And a part for which the quality check has not yet been completed, in the directory `./recent/`.

The folder `./timeseries_overview/` contains information about long time series.

---

## Point of contact

Deutscher Wetterdienst

CDC - Vertrieb Klima und Umwelt

Frankfurter Straße 135

63067 Offenbach

Tel: +49 (0) 69 8062-4400

Fax: +49 (0) 69 8062-4499

E-Mail: klima.vertrieb@dwd.de

---

## Dataset description

- **Unit(s):** numerical code
- **Statistical processing:** hourly value, time series
- **Temporal coverage:** 1949-01-01 -- ...
- **Spatial coverage:** stations in Germany
- **Projection:** WGS 84 (EPSG:4326)

**Format description**

In the folder `recent/` for each station a zip-archive is provided. The zip-archive contains the data and meta information about the station, instruments and algorithms.

The naming schema of the zip-archives is: `*_{product_code}_{station_id}_{begin_date}_{end_date}_hist.zip` (historical) or `*_{product_code}_{station_id}_akt.zip` (recent).

The file `WW_Stundenwerte_Beschreibung_Stationen.txt` contains information on the recent geographical position and the temporal data coverage per station.

The file `Wetter_Beschreibung.txt` has the definition of the weather codes and their meaning.

In the folder `./timeseries_overview`, information on long time series is available. The files provided (`TimeSeries_[DataType]_[Interval]_GE_[XXXYears]_[Parameter].html` or `*.txt`) contain a sorted overview of stations for which time series of >=100, >=50 and >=30 years are available. Information on the proportion of missing values is also provided.

### Content description

- `Stations_id` := Identifier of the station
- `Start` := Start date of the time series
- `End` := End date of the time series
- `Number_years` := Number of years of measurement operation
- `Missing_Years` := Number of missing years of measurement operation
- `Missing_values` := Number of missing values
- `max(Missing_period)>=25` := More than 25 years missing in the time series: indication of start date and end date
- `Station name` := Station name of the current location
- `Federal state` := Name of the federal state

> Translated with www.DeepL.com/Translator (free version)

### Application schema / CSV dialect

- **Delimiter:** `;`
- **Line terminator:** `\r\n`
- **Header:** true
- **Quote char:** `"`

CSV content description (columns):

- `STATIONS_ID` — Station ID (VARCHAR2)
- `MESS_DATUM` — reference date (CHAR, format: YYYYMMDDHH24)
- `QN_8` — quality level (NUMBER)
- `WW` — weather code (NUMBER)
- `WW_TEXT` — weather description (VARCHAR2)

---

## Quality information

The `QUALITAETS_NIVEAU` (QN) shows the quality control procedure applied for a data report (of several parameters) for a certain reporting time.

Data before and including 1980 can reach as best quality check level `QN=5`. Data after 1980 can reach `QN=10` as best quality check level.

QN levels:

- `QN = 1` : only formal control
- `QN = 2` : controlled with individually defined criteria
- `QN = 3` : automatic control and correction
- `QN = 5` : historic, subjective procedures
- `QN = 7` : second control done, before correction
- `QN = 8` : quality control outside ROUTINE
- `QN = 9` : not all parameters corrected
- `QN = 10` : quality control finished, all corrections finished

The `QUALITAETS_BYTE` (QB) denotes whether the value was objected to and/or corrected:

- `QB = 0` : not flagged
- `QB = 1` : had no objections (checked and not objected, or not checked and not objected)
- `QB = 2` : corrected
- `QB = 3` : confirmed with objection rejected
- `QB = 4` : added or calculated
- `QB = 5` : objected
- `QB = 6` : only formally checked
- `QB = 7` : formal objection
- `QB = -999` : quality flag does not exist

---

## Data origin

The data are taken from the station measuring networks of Deutscher Wetterdienst as well as its predecessor organisations. The dataset is regularly updated with recent as well as with recovered historical data.

From 1997 onwards, the data have been imported operationally into the central specialist database and archived (see Behrendt et al., 2011, and Kaspar et al., 2013). When going back to historical times, guidelines on observation procedure, instruments and observation times were issued by the authority in charge and might be incompletely recorded in the station metadata.

Between the end of the nineties and 2009 many stations were changed from manual to automated.

© Deutscher Wetterdienst 2024

---

## Resource maintenance

- In `recent/` the data files are updated daily. On a rolling basis, the data of the last 500 days (up to yesterday) are exchanged. Quality control has not yet been completed for these data, so values may change.
- In `historical/` the data files are updated annually. Quality control has been completed for this data, so values for the version are constant. During the annual version change, corrections and historical additions are incorporated.

---

## Validation and uncertainty estimate

The quality check and uncertainty assessment are explained in Kaspar et al., 2013. Different stages of quality control are run depending on the age of the data. In addition to manual quality control, automatic tests check completeness, temporal and spatial consistency, and compare against statistical thresholds (QualiMet software, Spengler, 2002).

## Uncertainties

Sources of long-term uncertainty include:

1. Changes in station height when a station was re-located (see station zip-files: `Metadaten_Geographie*`).
2. Changes in instrumentation (see `Metadaten_Geraete*`).
3. Varying quality control procedures.
4. Errors during data transfer or in software.
5. Change of observing personnel.
6. Other factors (see Freydank, 2014).

---

## Considerations for applications

When using the `historical/` and `recent/` directories together, account for temporal overlap and differing quality control procedures. When investigating long-term changes or trends, consult the `Uncertainties` section.

---

## Additional information

For the most recent data the quality control is not completed yet. There are still issues to be discovered in the historical data. Contributions to improve the data basis are welcome (see contact).

---

## Literature

- Behrendt, J., et al.: Beschreibung der Datenbasis des NKDZ. Version 3.5, Offenbach, 15.02.2011.
- DWD Vorschriften und Betriebsunterlagen Nr. 2 (VuB 2), Wetterschlüsselhandbuch Band D, Nov 2013.
- DWD Vorschriften und Betriebsunterlagen Nr. 3 (VuB 3), Beobachterhandbuch (BHB) für Wettermeldestellen des synoptisch-klimatologischen Mess- und Beobachtungsnetzes, März 2014a.
- DWD Vorschriften und Betriebsunterlagen Nr. 3 (VuB 3), Technikerhandbuch (THB) für Wettermeldestellen des synoptisch-klimatologischen Mess- und Beobachtungsnetzes, März 2014b.
- Freydank, E.: 150 Jahre staatliche Wetter- und Klimabeobachtungen in Sachsen. Tharandter Klimaprotokolle Band 21, 2014.
- Kaspar, F., et al.: Monitoring of climate change in Germany – data, products and services of Germany`s National Climate Data Centre. Adv. Sci. Res., 10, doi:10.5194/asr-10-99-2013, 99–106, 2013.
- Spengler, R.: The new Quality Control- and Monitoring System of the Deutscher Wetterdienst. Proceedings of the WMO Technical Conference on Meteorological and Environmental Instruments and Methods of Observation, Bratislava, 2002.

---

## Copyright

The Creative Commons BY 4.0 - Licence (CC BY 4.0) apply.

## Revision history

This document is maintained by Deutscher Wetterdienst, CDC - Betrieb, last edited at 2024-05-06.

© Deutscher Wetterdienst 2024
