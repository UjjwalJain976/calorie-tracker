import json
import os
import unittest
from datetime import datetime, timezone
from html.parser import HTMLParser
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import openai

# Importing the existing app initializes tables. Block that call in tests.
with patch("psycopg2.connect", side_effect=RuntimeError("Database disabled in tests")):
    import app as tracker


NUTRITION = {"calories": 120, "protein": 3, "carbs": 24, "fat": 1,
             "ai_confidence": "medium"}


class EstimationTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-only"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.sdk = patch.object(tracker, "OpenAI")
        self.create = self.sdk.start().return_value.responses.create
        self.addCleanup(self.sdk.stop)
        self.create.return_value = SimpleNamespace(
            status="completed", output_text=json.dumps(NUTRITION), output=[])

    def test_success_and_request_contract(self):
        with patch.dict(os.environ, {"OPENAI_MODEL": "gpt-4o-mini"}):
            self.assertEqual(tracker.estimate_calories("one banana"), NUTRITION)
        request = self.create.call_args.kwargs
        self.assertEqual(request["model"], "gpt-5.4-mini")
        self.assertEqual(set(request), {"model", "input", "text"})
        self.assertTrue(request["text"]["format"]["strict"])
        self.assertEqual(set(request["text"]["format"]["schema"]["required"]), set(NUTRITION))

    def test_invalid_payloads_fail_instead_of_becoming_zero(self):
        payloads = ["", "not json", "null", "[]", "{}"]
        for field in NUTRITION:
            data = dict(NUTRITION)
            del data[field]
            payloads.append(json.dumps(data))
        for value in [None, True, "123", -1, float("nan"), float("inf")]:
            payloads.append(json.dumps(dict(NUTRITION, calories=value)))
        payloads.append(json.dumps(dict(NUTRITION, ai_confidence="error")))
        for payload in payloads:
            with self.subTest(payload=payload), self.assertLogs(tracker.app.logger, "ERROR"):
                self.create.return_value.output_text = payload
                with self.assertRaises(tracker.CalorieEstimationError):
                    tracker.estimate_calories("food")

    def test_incomplete_and_refused_responses_fail(self):
        for status in ["incomplete", "failed", "queued"]:
            with self.subTest(status=status), self.assertLogs(tracker.app.logger, "ERROR"):
                self.create.return_value.status = status
                with self.assertRaises(tracker.CalorieEstimationError):
                    tracker.estimate_calories("food")
        self.create.return_value.status = "completed"
        self.create.return_value.output = [SimpleNamespace(content=[SimpleNamespace(type="refusal")])]
        with self.assertLogs(tracker.app.logger, "ERROR"), self.assertRaises(tracker.CalorieEstimationError):
            tracker.estimate_calories("food")

    def test_missing_key_fails_without_call(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}), self.assertLogs(tracker.app.logger, "ERROR"):
            with self.assertRaises(tracker.CalorieEstimationError):
                tracker.estimate_calories("food")
        self.create.assert_not_called()

    def test_api_errors_are_safely_logged(self):
        secret = "sk-private postgresql://user:password@private/db"
        for cls, status, code in [(openai.AuthenticationError, 401, "invalid_api_key"),
                                  (openai.PermissionDeniedError, 403, None),
                                  (openai.NotFoundError, 404, "model_not_found"),
                                  (openai.RateLimitError, 429, "insufficient_quota"),
                                  (openai.BadRequestError, 400, "unsupported_parameter")]:
            with self.subTest(error=cls.__name__):
                self.create.side_effect = cls(secret, response=httpx.Response(
                    status, request=httpx.Request("POST", "https://api.openai.com/v1/responses")),
                    body={"code": code, "param": "temperature", "message": secret})
                with self.assertLogs(tracker.app.logger, "ERROR") as logs:
                    with self.assertRaises(tracker.CalorieEstimationError):
                        tracker.estimate_calories("food")
                output = " ".join(logs.output)
                self.assertIn(cls.__name__, output)
                self.assertNotIn("sk-private", output)
                self.assertNotIn("postgresql", output)
                if code:
                    self.assertIn(code, output)

    def test_valid_zero_is_not_treated_as_failure(self):
        self.create.return_value.output_text = json.dumps(dict(NUTRITION, calories=0))
        self.assertEqual(tracker.estimate_calories("water")["calories"], 0)

    def test_quota_error_without_sdk_code_is_safely_identified(self):
        self.create.side_effect = openai.RateLimitError(
            "insufficient_quota: billing sk-private postgresql://secret/db",
            response=httpx.Response(429, request=httpx.Request("POST", "https://api.openai.com/v1/responses")),
            body=None)
        with self.assertLogs(tracker.app.logger, "ERROR") as logs:
            with self.assertRaises(tracker.CalorieEstimationError):
                tracker.estimate_calories("food")
        output = " ".join(logs.output)
        self.assertIn("insufficient_quota", output)
        self.assertNotIn("sk-private", output)
        self.assertNotIn("postgresql", output)


class SaveFlowTests(unittest.TestCase):
    def setUp(self):
        for name, result in [("fetch_food_entries_by_date", []), ("calculate_daily_totals", NUTRITION)]:
            mock = patch.object(tracker, name, return_value=result)
            mock.start()
            self.addCleanup(mock.stop)
        self.client = tracker.app.test_client()
        self.form = {"entry_date": "2026-11-01", "entry_time": "12:15",
                     "food_description": "one banana <script>"}

    @patch.object(tracker, "calculate_daily_totals", return_value=dict(NUTRITION))
    @patch.object(tracker, "fetch_food_entries_by_date", return_value=[])
    @patch.object(tracker, "insert_food_entry")
    @patch.object(tracker, "estimate_calories", side_effect=tracker.CalorieEstimationError)
    def test_failure_does_not_insert_and_preserves_retry(self, estimate, insert, entries, totals):
        response = self.client.post("/", data=self.form)
        self.assertEqual(response.status_code, 502)
        insert.assert_not_called()
        html = response.get_data(as_text=True)
        for text in ["Nothing was saved", "Estimate and Save", "2026-11-01", "12:15",
                     "one banana &lt;script&gt;"]:
            self.assertIn(text, html)
        # A retry with a valid response saves exactly once.
        estimate.side_effect = None
        estimate.return_value = NUTRITION
        self.assertEqual(self.client.post("/", data=self.form).status_code, 302)
        insert.assert_called_once()

    @patch.object(tracker, "insert_food_entry")
    @patch.object(tracker, "estimate_calories", return_value=NUTRITION)
    def test_success_saves_nutrition(self, estimate, insert):
        self.assertEqual(self.client.post("/", data=self.form).status_code, 302)
        insert.assert_called_once_with("2026-11-01", "12:15", self.form["food_description"],
                                       120, 3, 24, 1, "medium")

    @patch.object(tracker, "insert_food_entry")
    @patch.object(tracker, "estimate_calories")
    def test_empty_description_does_not_call_ai_or_save(self, estimate, insert):
        self.client.post("/", data=dict(self.form, food_description=" "))
        estimate.assert_not_called()
        insert.assert_not_called()


class DateInputParser(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.dates = {}
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "input" and attrs.get("type") == "date":
            self.dates[attrs["name"]] = attrs.get("value")


class FoodDateTests(unittest.TestCase):
    def setUp(self):
        self.client = tracker.app.test_client()
        self.real_insert = tracker.insert_food_entry
        for name, result in [("fetch_food_entries_by_date", []),
                             ("calculate_daily_totals", NUTRITION),
                             ("estimate_calories", NUTRITION), ("insert_food_entry", None)]:
            mock = patch.object(tracker, name, return_value=result)
            setattr(self, name, mock.start())
            self.addCleanup(mock.stop)
        self.form = {"entry_date": "2026-11-01", "entry_time": "12:15", "food_description": "banana"}

    def assert_date(self, response, selected_date):
        html = response.get_data(as_text=True)
        self.assertEqual(DateInputParser(html).dates, {"entry_date": selected_date})
        self.assertEqual(html.count('type="date"'), 1)
        self.assertIn("Entries for " + selected_date, html)
        self.assertNotIn('>View</button>', html)

    def test_posted_date_wins_over_old_url_and_reaches_insert_and_summary(self):
        self.insert_food_entry.side_effect = self.real_insert
        with patch.object(tracker, "get_db_connection") as connection:
            response = self.client.post("/?entry_date=2026-06-01", data=self.form, follow_redirects=True)
            cursor = connection.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value
            self.assertEqual(cursor.execute.call_args.args[1][0], "2026-11-01")
        self.assert_date(response, "2026-11-01")
        self.fetch_food_entries_by_date.assert_called_with("2026-11-01")
        self.calculate_daily_totals.assert_called_with("2026-11-01")

    def test_june_and_november_records_and_all_totals_follow_shared_date(self):
        for selected, calories in [("2026-06-01", 600), ("2026-11-01", 2000)]:
            with self.subTest(date=selected):
                self.fetch_food_entries_by_date.return_value = [dict(
                    NUTRITION, id=7, entry_date=selected, entry_time="12:00",
                    food_description="Food for " + selected)]
                self.calculate_daily_totals.return_value = dict(calories=calories, protein=31, carbs=42, fat=13)
                response = self.client.get("/?entry_date=" + selected)
                self.assert_date(response, selected)
                self.fetch_food_entries_by_date.assert_called_with(selected)
                self.calculate_daily_totals.assert_called_with(selected)
                html = response.get_data(as_text=True)
                for value in ["Food for " + selected, f"<strong>{calories}</strong>",
                              "<strong>31.0 g</strong>", "<strong>42.0 g</strong>",
                              "<strong>13.0 g</strong>", f"<strong>{abs(1800-calories)} kcal</strong>"]:
                    self.assertIn(value, html)
                self.assertIn("Over limit by" if calories > 1800 else "Calories left today", html)
                self.assertIn("selected_date=" + selected, html)
        self.insert_food_entry.assert_not_called()
        self.estimate_calories.assert_not_called()

    def test_today_changes_at_indian_midnight_without_restart(self):
        for instant, today in [(datetime(2026, 11, 1, 18, 29, 59, tzinfo=timezone.utc), "2026-11-01"),
                               (datetime(2026, 11, 1, 18, 30, tzinfo=timezone.utc), "2026-11-02")]:
            with self.subTest(instant=instant), patch.object(tracker, "datetime") as clock:
                clock.now.side_effect = lambda tz: instant.astimezone(tz)
                self.assert_date(self.client.get("/"), today)
                self.assertEqual(clock.now.call_args.args[0].key, "Asia/Kolkata")

    def test_historical_page_cleans_reload_url_and_root_uses_today(self):
        response = self.client.get("/?entry_date=2026-06-01")
        self.assert_date(response, "2026-06-01")
        self.assertIn('window.history.replaceState(null, \'\', "/")', response.get_data(as_text=True))
        # The browser reloads the URL set by replaceState, not the historical URL.
        with patch.object(tracker, "datetime") as clock:
            clock.now.return_value = datetime(2026, 11, 1, 12)
            response = self.client.get("/")
        self.assert_date(response, "2026-11-01")
        self.fetch_food_entries_by_date.assert_called_with("2026-11-01")
        self.calculate_daily_totals.assert_called_with("2026-11-01")

    def test_historical_entry_is_not_replaced_by_today(self):
        response = self.client.post("/", data=dict(self.form, entry_date="2020-02-29"), follow_redirects=True)
        self.assertEqual(self.insert_food_entry.call_args.args[0], "2020-02-29")
        self.assert_date(response, "2020-02-29")

    def test_ai_database_and_validation_failures_preserve_selected_date(self):
        self.estimate_calories.side_effect = tracker.CalorieEstimationError
        response = self.client.post("/?entry_date=2026-06-01", data=self.form)
        self.assertEqual(response.status_code, 502)
        self.assert_date(response, "2026-11-01")
        self.insert_food_entry.assert_not_called()
        self.estimate_calories.side_effect = None
        self.insert_food_entry.side_effect = RuntimeError("Database unavailable")
        response = self.client.post("/", data=self.form)
        self.assertEqual(response.status_code, 500)
        self.assert_date(response, "2026-11-01")
        response = self.client.post("/", data=dict(self.form, food_description=""))
        self.assertEqual(response.status_code, 400)
        self.assert_date(response, "2026-11-01")

    def test_invalid_html_date_is_not_saved_or_sent_to_ai(self):
        for value in ["01/11/2026", "2026-02-30", "20261101"]:
            with self.subTest(value=value):
                response = self.client.post("/", data=dict(self.form, entry_date=value))
                self.assertEqual(response.status_code, 400)
                self.assertNotEqual(self.fetch_food_entries_by_date.call_args.args[0], value)
        self.estimate_calories.assert_not_called()
        self.insert_food_entry.assert_not_called()

    def test_delete_success_and_failure_preserve_selected_date(self):
        for error in [None, RuntimeError("Database unavailable")]:
            with patch.object(tracker, "delete_food_entry", side_effect=error) as delete:
                response = self.client.post("/food/delete/7", data={
                    "entry_date": "2026-06-01"}, follow_redirects=True)
            delete.assert_called_once_with(7)
            self.assert_date(response, "2026-06-01")

    def test_edit_preserves_viewing_date_even_when_record_date_changes(self):
        entry = dict(NUTRITION, id=7, entry_date="2026-06-01", entry_time="12:00", food_description="banana")
        with patch.object(tracker, "get_food_entry_by_id", return_value=entry):
            response = self.client.get("/food/edit/7?selected_date=2026-06-01")
        self.assertIn('name="selected_date" value="2026-06-01"', response.get_data(as_text=True))
        form = dict(entry, entry_date="2026-11-01", selected_date="2026-06-01")
        with patch.object(tracker, "update_food_entry") as update:
            response = self.client.post("/food/edit/7", data=form, follow_redirects=True)
        self.assertEqual(update.call_args.args[1], "2026-11-01")
        self.assert_date(response, "2026-06-01")
        with patch.object(tracker, "update_food_entry", side_effect=RuntimeError("Database unavailable")):
            response = self.client.post("/food/edit/7", data=form)
        self.assertEqual(response.status_code, 500)
        self.assertIn('name="selected_date" value="2026-06-01"', response.get_data(as_text=True))
        response = self.client.post("/food/edit/7", data=dict(form, food_description=""))
        self.assertEqual(response.status_code, 400)
        self.assertIn('name="selected_date" value="2026-06-01"', response.get_data(as_text=True))


class SDKCompatibilityTests(unittest.TestCase):
    def test_pinned_sdk_serializes_request_and_parses_output(self):
        def handle(request):
            body = json.loads(request.content)
            self.assertEqual(body["model"], "gpt-5.4-mini")
            self.assertEqual(body["text"]["format"]["type"], "json_schema")
            return httpx.Response(200, json={
                "id": "resp_test", "created_at": 0, "model": "gpt-5.4-mini",
                "object": "response", "status": "completed", "error": None,
                "output": [{"id": "msg_test", "type": "message", "role": "assistant",
                            "status": "completed", "content": [{"type": "output_text",
                            "annotations": [], "text": json.dumps(NUTRITION)}]}],
            })
        with openai.OpenAI(api_key="sk-test-only", http_client=httpx.Client(
                transport=httpx.MockTransport(handle))) as client:
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-only"}), patch.object(
                    tracker, "OpenAI", return_value=client):
                self.assertEqual(tracker.estimate_calories("banana"), NUTRITION)


if __name__ == "__main__":
    unittest.main()
