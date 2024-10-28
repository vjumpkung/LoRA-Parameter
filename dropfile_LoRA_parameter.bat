@echo off
cd /d %~dp0
LoRA_parameter.py -i "%~1"
pause