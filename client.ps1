# Запуск спутника: микрофон и колонка здесь, мозги на ПК.
#   .\client.ps1                        - адрес сервера из config.yaml
#   .\client.ps1 192.168.31.35:8765     - адрес руками
#   .\client.ps1 --voice                - по клавише, без кодовой фразы
#
# На самом ПК сервер поднимается командой .\alisa.ps1 --serve
#
# Перед запуском код подтягивается с сервера (assistant\update.py).
# Код выхода 3 - сервер обновился посреди работы: обновиться и заново.
$env:PYTHONIOENCODING = "utf-8"
$python = "$PSScriptRoot\.venv\Scripts\python.exe"
do {
    & $python -m assistant.update @args
    & $python -m assistant.cli --client @args --auto-update
} while ($LASTEXITCODE -eq 3)
