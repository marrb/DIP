# Spracovanie dát

Táto časť repozitára obsahuje skripty na spracovanie vstupných videí, prípravu dát pre trénovanie a vyhodnocovanie modelu, ako aj spracovanie výsledkov užívateľskej štúdie.

## Prehľad skriptov

| Skript / Notebook                      | Popis |
|----------------------------------------|-------|
| `video_prep/prep_video.py`             | Rozdelenie videí na snímky, výber požadovaných dĺžok sekvencií. |
| `video_prep/video_prepper.py`          | Trieda poskytujúca metódy pre skript prep_video.py. |
| `video_processing/process_video.py`    | Extrakcia požadovaného počtu snímok z videa. |
| `video_processing/VideoProcessor.py`   | Trieda obsahujúca metódy pre manipuláciu so snímkami a videami. |
| `user_study_results_processing/QuestionProcessor.py` | Načítanie a spracovanie exportovaných odpovedí z dotazníka. |
| `user_study_results_processing/process_questions.ipynb` | Analýza výsledkov užívateľskej štúdie. |
| `user_study_results_processing/objective_metrics.ipynb` | Výpočet objektívnych metrík (CLIP, LPIPS) pre modely. |

## Spustenie

Tento projekt bol vyvíjaný a testovaný s **Python 3.9**. Odporúča sa použiť túto verziu kvôli kompatibilite knižníc.

Najprv nainštalujte potrebné knižnice pomocou `requirements.txt`:

```bash
pip install -r ./requirements.txt
```

Následne je možné skripty spustit príkazom:
```bash
python3 názov_skriptu.py [argumenty]
```

### Príklad

Príklad spustenia skriptu môže byť napríklad:
```bash
python3 ./video_prep/prep_video.py --video_dir ./video_prep/Dataset --output ./video_prep/Output
```

# Autor
- Martin Bublavý [xbubla02]