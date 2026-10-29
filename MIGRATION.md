# Переезд сервера на отдельный ПК с Ubuntu

План от 29.10.2026. **Ещё не выполнялся**: на Linux проект не запускался
ни разу. Всё, что ниже помечено «не проверено», проверять первым.

Цель: сервер (`--serve`), Telegram-бот и служба напоминаний живут на новом
ПК круглосуточно. Рабочий ПК с 5060 Ti в схеме не участвует и может быть
выключен. Спутник на ноутбуке не меняется.

## Новый ПК

- i5-10400F, 16 ГБ, NVMe 256 ГБ, GTX 1650 4 ГБ (Turing).
- Видеокарта нужна только распознаванию (whisper). Silero работает на
  процессоре (`tts.device: cpu`), DeepSeek и поиск — в интернете.
- GTX 1650 по требованиям подходит: NVIDIA, Turing и новее, от 4 ГБ
  (модель `large-v3-turbo` в int8 — около полутора гигабайт). **Скорость
  не замерена**: ожидаю 1–2 с на фразу против 0,3–0,5 с на 5060 Ti.
  Порог решения и запасные варианты — в шаге 9.

## Что где окажется

| Процесс | Команда | Порт |
|---|---|---|
| Сервер | `python -m assistant.cli --serve` | 8765 (домашняя сеть) |
| Wyoming (когда включим) | внутри сервера | 10300, 10200 |
| Telegram-бот | `python -m assistant.telegram_bot` | входящих нет |
| Напоминания | `python -m assistant.reminders` | 8766, только `127.0.0.1` |

Все три на одной машине, поэтому служба напоминаний наружу не слушает,
а сервер и бот ходят к ней и друг к другу через `127.0.0.1`.

## Шаг 1. Машина

1. Ubuntu 24.04 LTS. **Одна система, без второй рядом**: после
   отключения света загрузчик не должен выбрать Windows. Если вторая
   система всё же будет — в GRUB по умолчанию Ubuntu.
2. BIOS (ASUS PRIME B560): Advanced → APM → **Restore AC Power Loss →
   Power On**. Пропал свет — ПК поднялся сам.
3. Сон выключить:
   ```sh
   sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
   ```
4. Часовой пояс — время в промпт берётся у системы:
   ```sh
   sudo timedatectl set-timezone Europe/Moscow
   ```
5. SSH, чтобы работать с основного ПК:
   ```sh
   sudo apt install openssh-server git curl libportaudio2
   ```
   `libportaudio2` нужен `sounddevice`: на Windows библиотека лежит
   внутри колеса, на Linux — нет.
6. В роутере закрепить за ПК постоянный адрес. Проще всего —
   **`192.168.31.35`**, когда старый сервер выключен: тогда
   `config.local.yaml` на спутнике трогать не надо.

## Шаг 2. Драйвер NVIDIA

```sh
sudo ubuntu-drivers install
sudo reboot
nvidia-smi
```

`nvidia-smi` должен показать GTX 1650 и версию CUDA не ниже 12.8 — под неё
собран torch (`cu128`). CUDA Toolkit ставить не надо: cuBLAS и cuDNN
приходят пакетами pip, как на Windows.

## Шаг 3. Код и окружение

```sh
sudo mkdir -p /opt/alisa_selfhosted && sudo chown $USER: /opt/alisa_selfhosted
git clone https://github.com/msnkv/alisa_selfhosted.git /opt/alisa_selfhosted
cd /opt/alisa_selfhosted
```

Если репозиторий приватный — завести на сервере ключ SSH и добавить его в
GitHub как **deploy key только на чтение**: сервер должен уметь `pull`, но
не `push`.

Python **3.10**, как на рабочем ПК. В Ubuntu 24.04 стоит 3.12, а наши
закреплённые версии на нём не проверялись. Ставим через `uv`:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.10 --seed .venv
.venv/bin/python -m pip install torch==2.11.0 torchaudio --index-url https://download.pytorch.org/whl/cu128
grep -v pywebrtc requirements.txt > /tmp/req-server.txt
.venv/bin/python -m pip install -r /tmp/req-server.txt -r requirements-telegram.txt
```

- **`pywebrtc-audio` выкидываем**: колёса у него только под Windows, а
  нужен он лишь голосовому «стоп» на спутнике. Не проверено, встанет ли
  он на Linux; проверять незачем.
- `requirements-reminders.txt` отдельно не нужен: всё из него уже есть в
  основном списке.

## Шаг 4. Перенести то, чего нет в git

С рабочего ПК, например через `scp` или WinSCP:

| Что | Зачем |
|---|---|
| `.env` | ключ DeepSeek, токен бота, разрешённые ID |
| `data/` | гараж, рецепты, бюджет, about, заметки |
| `models/` | whisper (1,6 ГБ), Silero, silero-vad, vosk |
| `logs/dialogue.jsonl` | журнал диалогов, по желанию |

**`models/` копировать, а не качать заново**: huggingface и alphacephei из
России открываются не всегда, на рабочем ПК модели скачались через VPN.
**`.env` и `data/` не пересылать через облака и мессенджеры.**

```sh
chmod 600 .env
```

В `.env` дописать секрет для службы напоминаний (нужен и серверу, и
службе, файл у них общий):

```sh
python3 -c "import secrets; print('REMINDERS_TOKEN=' + secrets.token_hex(32))" >> .env
```

`config.local.yaml` на сервере — новый, свой:

```yaml
reminders:
  url: "http://127.0.0.1:8766"
```

`server.url` не нужен: бот без него ходит на `localhost:8765`.
Токен сервера (`server.token`) стоит задать, раз машина в сети постоянно,
— и тогда тот же токен прописать на спутнике и для бота.

## Шаг 5. Видеокарта для whisper: `LD_LIBRARY_PATH`

**Главная ожидаемая грабля.** На Windows каталоги cuBLAS и cuDNN из pip
дописывает в `PATH` функция `_register_cuda_dlls` в `assistant/stt.py`, а
на Linux она сразу выходит. Без каталогов будет та же ошибка, что на
шаге 5 этапа 1: модель загрузилась, на первом распознавании «cublas не
найден». На Linux каталоги задаются переменной до старта процесса:

```sh
export LD_LIBRARY_PATH=/opt/alisa_selfhosted/.venv/lib/python3.10/site-packages/nvidia/cublas/lib:/opt/alisa_selfhosted/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib
```

В службе systemd она прописана ниже. Не проверено вживую.

## Шаг 6. Проверка руками

```sh
export PYTHONIOENCODING=utf-8
.venv/bin/python -m assistant.cli --selftest
.venv/bin/python -m assistant.cli --ask "где лежат болгарки?"
.venv/bin/python -m tests.run_questions
```

Это то же, что `.\alisa.ps1` и `.\test.ps1` на Windows: они только
выставляют кодировку и зовут питон из `.venv`.

## Шаг 7. Службы systemd

Готовые файлы лежат в `infra/`: `alisa-server.service`,
`alisa-telegram.service`, `alisa-reminders.service`. В каждом
`User=alisa` заменить на своего пользователя Ubuntu.

```sh
cd /opt/alisa_selfhosted
for unit in alisa-server alisa-telegram alisa-reminders; do
    sed "s/^User=alisa$/User=$USER/" infra/$unit.service | sudo tee /etc/systemd/system/$unit.service >/dev/null
done
mkdir -p state
sudo systemctl daemon-reload
sudo systemctl enable --now alisa-reminders alisa-server alisa-telegram
systemctl status alisa-server
journalctl -u alisa-server -f
```

- `Restart=always`: упал — через пять секунд поднялся. Стартуют при
  загрузке, входить в систему не нужно.
- В `alisa-server.service` прописан `LD_LIBRARY_PATH` из шага 5.
- Журнал systemd ротируется сам. `logs/assistant.log` по-прежнему растёт
  без ротации, но теперь это не единственный журнал.

## Шаг 8. Файрвол

```sh
sudo ufw allow OpenSSH
sudo ufw allow from 192.168.31.0/24 to any port 8765 proto tcp
sudo ufw allow from 192.168.31.0/24 to any port 10200 proto tcp
sudo ufw allow from 192.168.31.0/24 to any port 10300 proto tcp
sudo ufw enable
```

**SSH разрешить до `enable`**, иначе отрежешь себя от машины. Порт
напоминаний 8766 не открываем: служба слушает только `127.0.0.1`.

## Шаг 9. Переключение и замер

1. На рабочем ПК **остановить сервер и бота**. Две копии бота на одном
   токене мешают друг другу. Локальный голосовой режим (`--voice`,
   `--wake`) на рабочем ПК запускать можно, он ни с кем не делит токен.
2. Отдать новому ПК адрес `192.168.31.35` (или поменять `server.url` на
   спутнике), запустить службы.
3. Проверить: вопрос из Telegram, спутник с ноутбука, напоминание на пару
   минут вперёд — **первая живая доставка за всю историю проекта**.
4. **Замерить распознавание на GTX 1650** по журналу (задержка STT на
   каждую фразу), на нескольких живых фразах:
   - до 1,5 с — оставляем;
   - медленнее — пробуем `stt.model: medium` или `small` в
     `config.local.yaml` и проверяем качество на живых фразах;
   - не устроило и это — GTX 1660 Super у брата (оценка: в полтора-два
     раза быстрее 1650, не замерено) или б/у RTX 3060.

## Обновления: автоматически, в тишине, с откатом

Правим и коммитим на рабочем ПК, `git push` — и всё. Сервер раз в минуту
сам проверяет GitHub, дожидается тишины, забирает код, перезапускает
службы и, если новое не взлетело, откатывается. Заходить на машину не
нужно. Сделано 29.10.2026, **на Linux не проверено**: логика прогнана на
временном репозитории с заглушками вместо systemd, sudo и curl.

### Что подхватывается и как

- **`data/`** — перечитывается на лету, перезапуск не нужен вовсе. Это
  заложено с этапа 1.
- **Код** — только перезапуском. Подменить модули внутри работающего
  питона (`importlib.reload`) для нас ненадёжно: модули держат ссылки
  друг на друга, половина осталась бы старой. Поэтому перезапуск
  делается автоматически и в тишине.
- **Спутник** — сам, как и раньше: сервер перезапустился с новым кодом,
  спутник увидел новую версию и подтянул файлы.

### Один прогон `infra/alisa-update.sh`

1. `git fetch`. Нового коммита нет — выход.
2. **Ждёт тишины** по адресу сервера `GET /status`: нет идущего ответа,
   последний вопрос был не меньше `QUIET_S` (120 с) назад, и ни одна
   сессия не ждёт «подтверждаю» по заметке или напоминанию. Не тихо —
   выход, через минуту снова. Сервер лежит — обновляемся сразу: мешать
   некому, а новый коммит, может, его и чинит.
3. `git merge --ff-only`. Изменились `requirements*.txt` — `pip install`
   (без `pywebrtc-audio`).
4. Перезапуск трёх служб: напоминания, сервер, бот.
5. **Проверка**: сервер ответил на `/health` за `HEALTH_WAIT_S` (300 с,
   это прогрев моделей), потом `SETTLE_S` (30 с) все три службы живы и
   ни разу не перезапускались сами.
6. **Не взлетело — откат**: `git reset --hard` на прошлый коммит, старые
   зависимости, перезапуск. Коммит записывается в
   `state/update-bad-commit` и больше не ставится; следующий коммит с
   исправлением пройдёт как обычно.
7. Владельцу (первый ID в `TELEGRAM_ALLOWED_USER_ID`) приходит сообщение
   в Telegram: «обновилась до …» или «не взлетело, откачено». Выключается
   `NOTIFY=0`.

Что `/status` считает разговором: `/ask`, `/ask-stream`, `/forget`,
`/v1/chat/completions` и Wyoming. Опрос манифеста спутником, `/health` и
сам `/status` — нет, иначе сервер никогда не выглядел бы простаивающим.

### Цена

- Перезапуск сбрасывает сессии в памяти: историю разговоров и режим
  сказки. Неподтверждённые заметки и напоминания не теряются — при них
  сервер не считается тихим.
- Около десяти секунд прогрева моделей (по замерам на рабочем ПК; на
  новом не мерено) сервер не отвечает: заговоришь со спутником в эти
  секунды — он сервер не дозовётся.
- Откат ловит только поломку при старте и в первые полминуты. Если новый
  код падает на каком-то вопросе через час, `Restart=always` его поднимет,
  но откатывать никто не станет.

### Установка (один раз)

```sh
sudo apt install curl
cd /opt/alisa_selfhosted
sudo install -m 755 infra/alisa-update.sh /usr/local/bin/alisa-update
sudo cp infra/alisa-update.service infra/alisa-update.timer /etc/systemd/system/
sudo cp infra/alisa-update.conf.example /etc/alisa-update.conf   # по желанию
sudo systemctl daemon-reload
sudo systemctl enable --now alisa-update.timer
```

- **Скрипт ставится копией в `/usr/local/bin`, а не запускается из
  репозитория.** bash читает файл по ходу выполнения, и `git pull`
  посреди прогона подменил бы его на лету. Заодно плохой коммит не
  сломает сам откат. Обратная сторона: **поменяли `alisa-update.sh` —
  повторить `sudo install` руками**, сам себя он не обновляет. То же с
  файлами служб в `/etc/systemd/system/`.
- Служба работает от root (ей надо перезапускать службы), а `git` и
  `pip` зовёт от владельца `/opt/alisa_selfhosted`, с его `HOME` — там
  ключ для GitHub.

### Посмотреть и вмешаться

```sh
journalctl -u alisa-update -n 50          # что делало обновление
systemctl list-timers alisa-update        # когда следующий прогон
sudo systemctl start alisa-update         # прогнать прямо сейчас
sudo systemctl stop alisa-update.timer    # приостановить автообновление
```

Скрипт **сам ничего не делает**, если на сервере есть свои коммиты или
правки файлов из git: пишет в журнал «обновление вручную» и ждёт. Править
на сервере можно только то, чего нет в git: `data/`, `.env`,
`config.local.yaml`.

### Руками, если таймер выключен

```sh
cd /opt/alisa_selfhosted && git pull
sudo systemctl restart alisa-reminders alisa-server alisa-telegram
```
