.PHONY: run test stop

run:
	.venv\Scripts\python.exe run_app.py

test:
	.venv\Scripts\python.exe -m pytest tests/

stop:
	.venv\Scripts\python.exe -m ts.model_server --stop
