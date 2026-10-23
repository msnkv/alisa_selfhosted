@echo off
rem Запуск спутника из обычной командной строки (cmd).
rem В cmd файл .ps1 не запускается, а открывается в блокноте -
rem поэтому рядом с .ps1 лежит этот .bat, он делает то же самое.
rem
rem   client.bat                        адрес сервера из config.yaml
rem   client.bat 192.168.31.35:8765     адрес руками
rem   client.bat --voice                по клавише, без кодовой фразы
rem
rem На самом ПК сервер поднимается командой:  alisa.bat --serve
rem
rem Питон ищется сам, и это важно для спутника: на ПК всё живёт в
rem .venv рядом с проектом, а на ноуте зависимости могут стоять в
rem системном питоне. Проверяем не наличие питона, а наличие в нём
rem pyyaml - иначе найдётся не тот, и запуск упадёт на импорте.
setlocal
set PYTHONIOENCODING=utf-8

set "RUNNER="
if exist "%~dp0.venv\Scripts\python.exe" set "RUNNER=%~dp0.venv\Scripts\python.exe"
if not defined RUNNER for /f "delims=" %%p in ('py -3 -c "import yaml,sys;print(sys.executable)" 2^>nul') do set "RUNNER=%%p"
if not defined RUNNER for /f "delims=" %%p in ('python -c "import yaml,sys;print(sys.executable)" 2^>nul') do set "RUNNER=%%p"
if not defined RUNNER goto :nopython

"%RUNNER%" -m assistant.cli --client %*
goto :eof

:nopython
echo.
echo Не нашёл питон с зависимостями.
echo Поставь их той же командой, какой ставил питон, например:
echo    py -3 -m pip install -r requirements-client.txt
echo Проверить, в каком питоне они оказались:
echo    py -3 -c "import yaml, sys; print(sys.executable)"
exit /b 1
