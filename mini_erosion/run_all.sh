#!/bin/bash
python 01_pull.py
python 02_shore_unet.py
python 03_metrics.py
python 04_pcmci.py
python 05_build_report.py
