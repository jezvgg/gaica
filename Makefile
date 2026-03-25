build:
	cd solution && zip -r ../solution.zip ./*

run:
	uv run gaica-local-runner/game/web_port/main.py 2>&1 &\
	sleep .5 &&\
	uv run gaica-sample-bot/main.py 127.0.0.1 9001 2>&1 &\
	uv run gaica-sample-bot/main.py 127.0.0.1 9001 2>&1 \

stop:
	pkill -f gaica