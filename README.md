# Personal Daily Fitness Tracker

A small private Flask app for daily food, calories, macros, weight tracking, and simple reports.

## Features

- Add multiple food entries per day by date and time
- Estimate calories, protein, carbs, and fat with the OpenAI API
- Edit or delete food entries
- Show daily totals against an 1800 kcal limit
- Add weight and notes
- Show latest weight progress toward an 80 kg target
- View calorie and weight graphs with Chart.js
- Export food entries and weight history as CSV
- Store data in Supabase PostgreSQL

## Local Setup

1. Create a virtual environment:

```bash
python -m venv .venv
```

2. Activate it:

```bash
.venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create your `.env` file:

```bash
copy .env.example .env
```

5. Edit `.env`:

```env
OPENAI_API_KEY=your_real_openai_api_key
DATABASE_URL=your_supabase_database_url
```

6. Run the app:

```bash
python app.py
```

7. Open:

```text
http://localhost:5000
```

The PostgreSQL tables are created automatically in Supabase when the app starts.

## Main Routes

- `/` food tracker and daily calorie ledger
- `/food/edit/<id>` edit a food entry
- `/weight` weight log and progress
- `/reports/calories` calorie graph
- `/reports/weight` weight graph
- `/export/food.csv` food CSV export
- `/export/weight.csv` weight CSV export
- `/test-db` safe Supabase connection test

## Render Deployment

1. Push this project to GitHub.
2. Create a new Render Web Service.
3. Connect your GitHub repository.
4. Use these settings:

```text
Build Command: pip install -r requirements.txt
Start Command: gunicorn app:app
```

The included `.python-version` file pins Render to Python 3.11.9 so `psycopg2-binary` installs a compatible wheel. If Render still uses Python 3.14, add an environment variable named `PYTHON_VERSION` with value `3.11.9`, then redeploy with a cleared build cache.

5. Add environment variables in Render:

```env
OPENAI_API_KEY=your_real_openai_api_key
DATABASE_URL=your_supabase_database_url
```

6. Deploy.

## Supabase Database

Use the PostgreSQL connection string from Supabase Project Settings -> Database -> Connection string. Put that value in `DATABASE_URL`; do not write it directly in code.

## Notes

- Do not commit `.env`.
- No login or user system is included.
- SQLite is not used.
