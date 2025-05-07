# Model

Táto časť repozitára obsahuje upravenú verziu modelu Video-P2P s integrovanými architektonickými rozšíreniami STAM, FFAM a ďalšími temporálnymi modulmi. Cieľom úprav je zlepšiť časovú konzistenciu a kvalitu editácie videí na základe textového vstupu.

Model je postavený na základe **Stable Diffusion v1.5** a implementovaný v prostredí **Python 3.10.12** s podporou **CUDA 11.8**.


## Spustenie a konfigurácia

Pre správne spojazdnenie modelu postupujte podľa nasledujúcich krokov.

```bash
pip install -r requirements.txt
```

Na stiahnutie modelových váh je potrebné mať nainštalovaný Git LFS:
```bash
sudo apt-get update
sudo apt-get install git-lfs
git lfs install
```

Klonovanie modelu Stable Diffusion v1.5:
```bash
git clone https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
```

Pre správnu kompatibilitu je potrebné upraviť súbor:
``` bash
vim /usr/local/lib/python3.10/dist-packages/diffusers/dynamic_modules_utils.py
```
- Odstrániť **cache_downloads** z importov
- V kóde nahradiť za **hf_hub_download**

### Inštalácia xformers:
```bash
git clone https://github.com/facebookresearch/xformers/
cd xformers
git submodule update --init --recursive
sudo apt-get install python3-dev
```

## Ladenie a testovanie
Model sa ladí a testuje v dvoch fázach:

### 1. One-shot tuning (prispôsobenie modelu vstupnému videu)

Táto fáza slúži na inicializáciu váh tak, aby boli citlivé na konkrétne video a jeho obsah. Spúšťa sa cez konfiguračný súbor `*.yaml`, ktorý definuje cestu k dátam, parametre tréningu a ďalšie nastavenia:

```bash
python3 run_tuning.py --config="configs/<nazov>-tune.yaml"
```

### Argumenty

| Argument           | Popis |
|--------------------|--------|
| `--config`         | Cesta ku konfiguračnému `.yaml` súboru |
| `--model_type`     | Typ použitého modelu: `video_p2p`, `video_p2p_ei`,  `video_p2p_ei_plus`. (volitelný, predvolený: `video_p2p`) |

### 2. Spustenie upraveného modelu (úprava videa)
Po ladení je možné spustiť samotnú úpravu videa.

```bash
python3 run_videop2p.py --config="configs/<nazov>-p2p.yaml"
```

### Argumenty

| Argument           | Popis |
|--------------------|--------|
| `--config`         | Cesta ku konfiguračnému `.yaml` súboru |
| `--fast`           | Zapne rýchly režim (nižší počet difúznych krokov, vhodné na testovanie, volitelný). |
| `--model_type`     | Typ použitého modelu: `video_p2p`, `video_p2p_ei`,  `video_p2p_ei_plus`. (volitelný, predvolený: `video_p2p`) |

## Príklad spustenia

Ako príklad slúži video so skákajúcim zajacom, ktoré sa upravuje podľa textu „a origami rabbit is jumping on the grass“.

### Ladenie
```bash
python3 run_tuning.py --config="configs/rabbit-jump-tune.yaml" --model_type="video_p2p_ei"
```

### Spustenie
```bash
python3 run_videop2p.py --config="configs/rabbit-jump-p2p.yaml" --model_type="video_p2p_ei"
```

Upravené video sa uloží do priečinka:
```bash
./outputs/rabbit-jump/results
```

# Autor
- Martin Bublavý [xbubla02]
