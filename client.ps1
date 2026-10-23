# Запуск спутника: микрофон и колонка здесь, мозги на ПК.
#   .\client.ps1                        - адрес сервера из config.yaml
#   .\client.ps1 192.168.31.35:8765     - адрес руками
#   .\client.ps1 --voice                - по клавише, без кодовой фразы
#
# На самом ПК сервер поднимается командой .\alisa.ps1 --serve
$env:PYTHONIOENCODING = "utf-8"
& "$PSScriptRoot\.venv\Scripts\python.exe" -m assistant.cli --client @args
