# Web Port (Python + HTML/JS)

Этот модуль содержит web-порт игры с backend-симуляцией на Python и визуализацией в браузере.
Для обычной проверки участнических ботов используйте `run_local_runner.py`; `web_port` нужен в основном для низкоуровневой локальной отладки и визуализации.

## Запуск

Из корня репозитория или из корня `gaica-local-runner`:

```bash
python game/web_port/main.py --bot-port 9001 --web-port 8080
```

По умолчанию карта выбирается случайно для матча.

После старта:

- TCP для ботов: `127.0.0.1:9001`
- Web UI: `http://127.0.0.1:8080/`

## Запуск сразу с тестовыми ботами

Основная команда:

```bash
python game/web_port/main.py --with-test-bots
```

В этом режиме после завершения раунда автоматически стартует следующий через `1` секунду.

## Протокол TCP бота

Сервер шлёт JSON-lines:

- `{"type":"hello", ...}`
- `{"type":"round_start", ...}`
- `{"type":"tick", ...}`
- `{"type":"round_end", ...}`

Бот отправляет JSON-lines:

Сразу после подключения нужно отправить:

```json
{
  "type": "register",
  "name": "my-bot",
  "protocol": "standalone"
}
```

`web_port` использует `name` для отображения, а платформа GAICA требует `register` как обязательный первый шаг бот-протокола.

```json
{
  "type": "command",
  "seq": 42,
  "move": [1, 0],
  "aim": [1, 0],
  "shoot": false,
  "kick": false,
  "pickup": true,
  "drop": false,
  "throw": false
}
```
