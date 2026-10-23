@echo off
rem Запуск ассистента из обычной командной строки (cmd).
rem В cmd файл .ps1 не запускается, а открывается в блокноте -
rem поэтому рядом с alisa.ps1 лежит этот .bat, он делает то же самое.
rem
rem   alisa.bat --serve                     отвечать по домашней сети
rem   alisa.bat --ask "где болгарки?"       один вопрос
rem   alisa.bat --say "текст"               озвучить свой текст
rem   alisa.bat --wake                      по кодовой фразе
setlocal
set PYTHONIOENCODING=utf-8
"%~dp0.venv\Scripts\python.exe" -m assistant.cli %*
