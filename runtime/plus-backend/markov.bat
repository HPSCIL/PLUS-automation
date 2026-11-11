@echo off
cd /d "%~dp0"
cpp\PLUS.exe -t markov -i PLUS_Markov.tmp -o cpp/output/markov.csv
