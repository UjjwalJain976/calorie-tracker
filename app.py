import csv
import io
import json
import os
from datetime import date, datetime
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from flask import Flask, Response, flash, redirect, render_template, request, url_for
from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-this-for-local-use")

DAILY_CALORIE_LIMIT = 1800
TARGET_WEIGHT = 80


def get_db_connection():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is missing from environment.")

    return psycopg2.connect(database_url, cursor_factory=RealDictCursor)


def database_error_message(error):
    if isinstance(error, RuntimeError):
        return str(error)

    error_type = type(error).__name__
    raw_message = str(error).lower()

    if "password authentication failed" in raw_message:
        safe_message = "password authentication failed for the configured database user"
    elif "could not translate host name" in raw_message:
        safe_message = "database host could not be found"
    elif "connection timed out" in raw_message or "timeout expired" in raw_message:
        safe_message = "database connection timed out"
    elif "connection refused" in raw_message:
        safe_message = "database connection was refused"
    elif "ssl" in raw_message:
        safe_message = "database SSL connection failed"
    else:
        safe_message = "database connection failed; check DATABASE_URL and Supabase status"

    return f"{error_type}: {safe_message}."


def test_database_connection():
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 AS ok")
            return cur.fetchone()["ok"] == 1


def init_db():
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS food_entries (
                    id SERIAL PRIMARY KEY,
                    entry_date DATE NOT NULL,
                    entry_time TIME,
                    food_description TEXT NOT NULL,
                    calories NUMERIC DEFAULT 0,
                    protein NUMERIC DEFAULT 0,
                    carbs NUMERIC DEFAULT 0,
                    fat NUMERIC DEFAULT 0,
                    ai_confidence TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS weight_log (
                    id SERIAL PRIMARY KEY,
                    entry_date DATE NOT NULL,
                    weight NUMERIC NOT NULL,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )


def to_float(value, default=0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def empty_calorie_estimate():
    return {
        "calories": 0,
        "protein": 0,
        "carbs": 0,
        "fat": 0,
        "ai_confidence": "error",
    }


def estimate_calories(food_description):
    try:
        if not os.getenv("OPENAI_API_KEY"):
            return empty_calorie_estimate()

        client = OpenAI()

        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "calories": {"type": "number"},
                "protein": {"type": "number"},
                "carbs": {"type": "number"},
                "fat": {"type": "number"},
                "ai_confidence": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                },
            },
            "required": ["calories", "protein", "carbs", "fat", "ai_confidence"],
        }

        response = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            input=[
                {
                    "role": "system",
                    "content": (
                        "Estimate nutrition for food the user personally ate. "
                        "Return structured JSON only. Use calories in kcal and macros in grams. "
                        "If the user gives a quantity, use that quantity. "
                        "If quantity is missing, assume one normal household serving and set ai_confidence to low. "
                        "Do not assume 100g unless the user writes 100g."
                    ),
                },
                {"role": "user", "content": food_description},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "nutrition_estimate",
                    "strict": True,
                    "schema": schema,
                }
            },
        )

        data = json.loads(response.output_text)
        ai_confidence = data.get("ai_confidence")

        if ai_confidence not in ["low", "medium", "high"]:
            return empty_calorie_estimate()

        return {
            "calories": to_float(data.get("calories")),
            "protein": to_float(data.get("protein")),
            "carbs": to_float(data.get("carbs")),
            "fat": to_float(data.get("fat")),
            "ai_confidence": ai_confidence,
        }
    except Exception:
        return empty_calorie_estimate()


def insert_food_entry(entry_date, entry_time, food_description, calories, protein, carbs, fat, ai_confidence):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO food_entries (
                    entry_date, entry_time, food_description, calories,
                    protein, carbs, fat, ai_confidence
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    entry_date,
                    entry_time or None,
                    food_description,
                    calories,
                    protein,
                    carbs,
                    fat,
                    ai_confidence,
                ),
            )


def get_food_entry_by_id(entry_id):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    entry_date::text AS entry_date,
                    to_char(entry_time, 'HH24:MI') AS entry_time,
                    food_description,
                    calories,
                    protein,
                    carbs,
                    fat,
                    ai_confidence,
                    to_char(created_at, 'YYYY-MM-DD HH24:MI:SS') AS created_at
                FROM food_entries
                WHERE id = %s
                """,
                (entry_id,),
            )
            return cur.fetchone()


def update_food_entry(entry_id, entry_date, entry_time, food_description, calories, protein, carbs, fat, ai_confidence):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE food_entries
                SET entry_date = %s,
                    entry_time = %s,
                    food_description = %s,
                    calories = %s,
                    protein = %s,
                    carbs = %s,
                    fat = %s,
                    ai_confidence = %s
                WHERE id = %s
                """,
                (
                    entry_date,
                    entry_time or None,
                    food_description,
                    calories,
                    protein,
                    carbs,
                    fat,
                    ai_confidence,
                    entry_id,
                ),
            )


def fetch_food_entries_by_date(entry_date):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    entry_date::text AS entry_date,
                    to_char(entry_time, 'HH24:MI') AS entry_time,
                    food_description,
                    calories,
                    protein,
                    carbs,
                    fat,
                    ai_confidence,
                    to_char(created_at, 'YYYY-MM-DD HH24:MI:SS') AS created_at
                FROM food_entries
                WHERE entry_date = %s
                ORDER BY COALESCE(entry_time, created_at::time) ASC,
                         created_at ASC,
                         id ASC
                """,
                (entry_date,),
            )
            return cur.fetchall()


def calculate_daily_totals(entry_date):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    COALESCE(SUM(calories), 0) AS calories,
                    COALESCE(SUM(protein), 0) AS protein,
                    COALESCE(SUM(carbs), 0) AS carbs,
                    COALESCE(SUM(fat), 0) AS fat
                FROM food_entries
                WHERE entry_date = %s
                """,
                (entry_date,),
            )
            return cur.fetchone()


def delete_food_entry(entry_id):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM food_entries WHERE id = %s", (entry_id,))


def delete_weight_entry(entry_id):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM weight_log WHERE id = %s", (entry_id,))


def insert_weight_log(entry_date, weight, notes):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO weight_log (entry_date, weight, notes)
                VALUES (%s, %s, %s)
                """,
                (entry_date, weight, notes),
            )


def fetch_weight_history():
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    entry_date::text AS entry_date,
                    weight,
                    notes,
                    to_char(created_at, 'YYYY-MM-DD HH24:MI:SS') AS created_at
                FROM weight_log
                ORDER BY entry_date DESC, created_at DESC, id DESC
                """
            )
            return cur.fetchall()


def fetch_latest_weight():
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    entry_date::text AS entry_date,
                    weight,
                    notes,
                    to_char(created_at, 'YYYY-MM-DD HH24:MI:SS') AS created_at
                FROM weight_log
                ORDER BY entry_date DESC, created_at DESC, id DESC
                LIMIT 1
                """
            )
            return cur.fetchone()


def fetch_calorie_report(start_date, end_date):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    entry_date::text AS entry_date,
                    COALESCE(SUM(calories), 0) AS calories
                FROM food_entries
                WHERE entry_date BETWEEN %s AND %s
                GROUP BY entry_date
                ORDER BY entry_date ASC
                """,
                (start_date, end_date),
            )
            return cur.fetchall()


def fetch_weight_report(start_date, end_date):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    entry_date::text AS entry_date,
                    weight,
                    notes,
                    to_char(created_at, 'YYYY-MM-DD HH24:MI:SS') AS created_at
                FROM weight_log
                WHERE entry_date BETWEEN %s AND %s
                ORDER BY entry_date ASC, created_at ASC, id ASC
                """,
                (start_date, end_date),
            )
            return cur.fetchall()


@app.route("/test-db")
def test_db():
    dotenv_file_exists = (BASE_DIR / ".env").exists()
    database_url_is_set = bool(os.getenv("DATABASE_URL"))

    try:
        if test_database_connection():
            return Response("Database connection successful.", mimetype="text/plain")
    except Exception as error:
        details = [
            "Database connection failed.",
            f".env file found: {dotenv_file_exists}",
            f"DATABASE_URL loaded: {database_url_is_set}",
            f"Safe error: {database_error_message(error)}",
        ]
        return Response("\n".join(details), status=500, mimetype="text/plain")

    return Response(
        "Database connection failed.\nSafe error: Database test query did not return the expected result.",
        status=500,
        mimetype="text/plain",
    )


@app.route("/", methods=["GET", "POST"])
def index():
    selected_date = request.values.get("entry_date") or date.today().isoformat()

    if request.method == "POST":
        entry_time = request.form.get("entry_time", "").strip()
        food_description = request.form.get("food_description", "").strip()

        if not food_description:
            flash("Please type what you ate.")
            return redirect(url_for("index", entry_date=selected_date))

        if not entry_time:
            entry_time = datetime.now().strftime("%H:%M")

        nutrition = estimate_calories(food_description)
        if nutrition["ai_confidence"] == "error":
            flash("OpenAI estimate failed, so this entry was saved with 0 values.")

        try:
            insert_food_entry(
                selected_date,
                entry_time,
                food_description,
                nutrition["calories"],
                nutrition["protein"],
                nutrition["carbs"],
                nutrition["fat"],
                nutrition["ai_confidence"],
            )
        except Exception as error:
            flash(database_error_message(error))
            return redirect(url_for("index", entry_date=selected_date))

        flash("Food entry saved.")
        return redirect(url_for("index", entry_date=selected_date))

    try:
        entries = fetch_food_entries_by_date(selected_date)
        totals = calculate_daily_totals(selected_date)
    except Exception as error:
        flash(database_error_message(error))
        entries = []
        totals = {"calories": 0, "protein": 0, "carbs": 0, "fat": 0}

    calories_consumed = float(totals["calories"] or 0)
    calories_left = DAILY_CALORIE_LIMIT - calories_consumed

    return render_template(
        "index.html",
        entries=entries,
        selected_date=selected_date,
        totals=totals,
        daily_calorie_limit=DAILY_CALORIE_LIMIT,
        calories_left=calories_left,
    )


@app.route("/food/edit/<int:entry_id>", methods=["GET", "POST"])
def edit_food_entry(entry_id):
    if request.method == "POST":
        entry_date = request.form.get("entry_date") or date.today().isoformat()
        entry_time = request.form.get("entry_time", "").strip()
        food_description = request.form.get("food_description", "").strip()
        ai_confidence = request.form.get("ai_confidence", "").strip()

        if not food_description:
            flash("Please type what you ate.")
            return redirect(url_for("edit_food_entry", entry_id=entry_id))

        if ai_confidence not in ["low", "medium", "high", "error"]:
            ai_confidence = "error"

        try:
            update_food_entry(
                entry_id,
                entry_date,
                entry_time,
                food_description,
                to_float(request.form.get("calories")),
                to_float(request.form.get("protein")),
                to_float(request.form.get("carbs")),
                to_float(request.form.get("fat")),
                ai_confidence,
            )
        except Exception as error:
            flash(database_error_message(error))
            return redirect(url_for("edit_food_entry", entry_id=entry_id))

        flash("Food entry updated.")
        return redirect(url_for("index", entry_date=entry_date))

    try:
        entry = get_food_entry_by_id(entry_id)
    except Exception as error:
        flash(database_error_message(error))
        return redirect(url_for("index"))

    if not entry:
        flash("Food entry not found.")
        return redirect(url_for("index"))

    return render_template("edit_food.html", entry=entry)


@app.route("/food/delete/<int:entry_id>", methods=["POST"])
def delete_food_entry_route(entry_id):
    selected_date = request.form.get("entry_date") or date.today().isoformat()

    try:
        delete_food_entry(entry_id)
        flash("Food entry deleted.")
    except Exception as error:
        flash(database_error_message(error))

    return redirect(url_for("index", entry_date=selected_date))


@app.route("/weight", methods=["GET", "POST"])
def weight():
    if request.method == "POST":
        entry_date = request.form.get("entry_date") or date.today().isoformat()
        weight_value = request.form.get("weight")
        notes = request.form.get("notes", "").strip()

        if not weight_value:
            flash("Please enter your weight.")
            return redirect(url_for("weight"))

        try:
            insert_weight_log(
                entry_date,
                to_float(weight_value),
                notes,
            )
        except Exception as error:
            flash(database_error_message(error))
            return redirect(url_for("weight"))

        flash("Weight log saved.")
        return redirect(url_for("weight"))

    try:
        logs = fetch_weight_history()
        latest_weight = fetch_latest_weight()
    except Exception as error:
        flash(database_error_message(error))
        logs = []
        latest_weight = None

    weight_left_to_reduce = None
    if latest_weight:
        weight_left_to_reduce = float(latest_weight["weight"] or 0) - TARGET_WEIGHT

    return render_template(
        "weight.html",
        today=date.today().isoformat(),
        logs=logs,
        latest_weight=latest_weight,
        target_weight=TARGET_WEIGHT,
        weight_left_to_reduce=weight_left_to_reduce,
    )


@app.route("/weight/delete/<int:entry_id>", methods=["POST"])
def delete_weight_entry_route(entry_id):
    try:
        delete_weight_entry(entry_id)
        flash("Weight entry deleted.")
    except Exception as error:
        flash(database_error_message(error))

    return redirect(url_for("weight"))


@app.route("/reports/calories")
def calorie_report():
    today_text = date.today().isoformat()
    start_date = request.args.get("start_date") or today_text
    end_date = request.args.get("end_date") or today_text

    try:
        rows = fetch_calorie_report(start_date, end_date)
    except Exception as error:
        flash(database_error_message(error))
        rows = []

    labels = [row["entry_date"] for row in rows]
    calories = [float(row["calories"] or 0) for row in rows]
    limit_values = [DAILY_CALORIE_LIMIT for _ in rows]

    return render_template(
        "calorie_report.html",
        start_date=start_date,
        end_date=end_date,
        rows=rows,
        labels=labels,
        calories=calories,
        limit_values=limit_values,
        daily_calorie_limit=DAILY_CALORIE_LIMIT,
    )


@app.route("/reports/weight")
def weight_report():
    today_text = date.today().isoformat()
    start_date = request.args.get("start_date") or today_text
    end_date = request.args.get("end_date") or today_text

    try:
        rows = fetch_weight_report(start_date, end_date)
    except Exception as error:
        flash(database_error_message(error))
        rows = []

    labels = [row["entry_date"] for row in rows]
    weights = [float(row["weight"] or 0) for row in rows]
    target_values = [TARGET_WEIGHT for _ in rows]

    return render_template(
        "weight_report.html",
        start_date=start_date,
        end_date=end_date,
        rows=rows,
        labels=labels,
        weights=weights,
        target_values=target_values,
        target_weight=TARGET_WEIGHT,
    )


@app.route("/export/food.csv")
def export_food_csv():
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        entry_date::text AS entry_date,
                        to_char(entry_time, 'HH24:MI') AS entry_time,
                        food_description,
                        calories,
                        protein,
                        carbs,
                        fat,
                        ai_confidence,
                        to_char(created_at, 'YYYY-MM-DD HH24:MI:SS') AS created_at
                    FROM food_entries
                    ORDER BY entry_date DESC,
                             COALESCE(entry_time, created_at::time) ASC,
                             created_at ASC,
                             id ASC
                    """
                )
                entries = cur.fetchall()
    except Exception as error:
        return Response(database_error_message(error), status=500, mimetype="text/plain")

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "entry_date",
            "entry_time",
            "food_description",
            "calories",
            "protein",
            "carbs",
            "fat",
            "ai_confidence",
            "created_at",
        ]
    )
    for row in entries:
        writer.writerow([row[key] for key in row.keys()])

    filename = f"food_entries_{date.today().isoformat()}.csv"
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.route("/export/weight.csv")
def export_weight_csv():
    try:
        rows = fetch_weight_history()
    except Exception as error:
        return Response(database_error_message(error), status=500, mimetype="text/plain")

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["entry_date", "weight", "notes", "created_at"])
    for row in rows:
        writer.writerow(
            [
                row["entry_date"],
                row["weight"],
                row["notes"],
                row["created_at"],
            ]
        )

    filename = f"weight_log_{date.today().isoformat()}.csv"
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


if __name__ == "__main__":
    try:
        init_db()
    except Exception as error:
        print(database_error_message(error))
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
else:
    try:
        init_db()
    except Exception:
        pass
