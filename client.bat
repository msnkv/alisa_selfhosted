@echo off
rem Запуск спутника из обычной командной строки (cmd).
rem Микрофон и колонка здесь, мозги на ПК.
rem
rem   client.bat                        адрес сервера из config.yaml
rem   client.bat 192.168.31.35:8765     адрес руками
rem   client.bat --voice                по клавише, без кодовой фразы
rem
rem На самом ПК сервер поднимается командой:  alisa.bat --serve
setlocal
set PYTHONIOENCODING=utf-8
"%~dp0.venv\Scripts\python.exe" -m assistant.cli --client %*
