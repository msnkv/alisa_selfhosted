# Запуск ассистента без длинного пути к python.exe.
#   .\alisa.ps1              - диалоговый режим
#   .\alisa.ps1 --selftest   - проверка окружения
#   .\alisa.ps1 --ask "где лежат болгарки?"
$env:PYTHONIOENCODING = "utf-8"
& "$PSScriptRoot\.venv\Scripts\python.exe" -m assistant.cli @args
