.PHONY: run test stop

run:
	cd story-continuation && .venv\Scripts\python.exe run_app.py

test:
	cd story-continuation && .venv\Scripts\python.exe -m pytest tests/

stop:
	cd story-continuation && .venv\Scripts\python.exe -m ts.model_server --stop
