@echo off
cd /d %~dp0
python LoRA_parameter.py -i "%~1"
pause