# Прогон тестовых вопросов из tests/questions.yaml.
#   .\test.ps1                 весь набор
#   .\test.ps1 --only fresh    только вопросы, требующие интернета
#   .\test.ps1 --repeat 3      каждый вопрос по три раза
$env:PYTHONIOENCODING = "utf-8"
& "$PSScriptRoot\.venv\Scripts\python.exe" -m tests.run_questions @args
