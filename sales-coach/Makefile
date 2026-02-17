run:
	cd .. && dotenvx run -f .env -- uv run uvicorn app:app --host 0.0.0.0 --port 5050 --reload --app-dir sales-coach 2>&1 | tee -a /tmp/sales_coach.log
