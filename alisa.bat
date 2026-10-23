@echo off
rem Запуск ассистента из обычной командной строки (cmd).
rem В cmd файл .ps1 не запускается, а открывается в блокноте -
rem поэтому рядом с .ps1 лежит этот .bat, он делает то же самое.
rem
rem   alisa.bat --serve                     отвечать по домашней сети
rem   alisa.bat --ask "где болгарки?"       один вопрос
rem   alisa.bat --say "текст"               озвучить свой текст
rem   alisa.bat --wake                      по кодовой фразе
rem   alisa.bat --mic-test                  цифры с микрофона
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

"%RUNNER%" -m assistant.cli %*
goto :eof

:nopython
echo.
echo Не нашёл питон с зависимостями.
echo Поставь их той же командой, какой ставил питон, например:
echo    py -3 -m pip install -r requirements-client.txt
echo Проверить, в каком питоне они оказались:
echo    py -3 -c "import yaml, sys; print(sys.executable)"
exit /b 1
