import sys
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from insights.ai_insights import AIInsightsEngine


@pytest.fixture
def engine():
    return AIInsightsEngine()


@pytest.fixture
def metrics():
    return {"MAE": 1200.0, "RMSE": 1800.0, "R2": 0.87, "MAPE": 4.5}


@pytest.fixture
def feature_importance_df():
    return pd.DataFrame({"feature": ["f1", "f2", "f3"], "importance": [0.5, 0.3, 0.2]})


# ---------------------------------------------------------------------------
# _call_pollinations
# ---------------------------------------------------------------------------

class TestCallPollinations:
    def test_success_returns_stripped_text(self, engine):
        mock_resp = MagicMock()
        mock_resp.text = "  Pollinations answer  "
        mock_resp.raise_for_status = MagicMock()
        with patch("insights.ai_insights.requests.post", return_value=mock_resp):
            text, err = engine._call_pollinations("prompt", "system")
        assert text == "Pollinations answer"
        assert err == ""

    def test_empty_response_returns_none(self, engine):
        mock_resp = MagicMock()
        mock_resp.text = "   "
        mock_resp.raise_for_status = MagicMock()
        with patch("insights.ai_insights.requests.post", return_value=mock_resp):
            text, err = engine._call_pollinations("prompt", "system")
        assert text is None
        assert "Empty" in err

    def test_connection_error_returns_none(self, engine):
        import requests
        with patch("insights.ai_insights.requests.post",
                   side_effect=requests.exceptions.ConnectionError("timeout")):
            text, err = engine._call_pollinations("prompt", "system")
        assert text is None
        assert len(err) > 0

    def test_http_error_returns_none(self, engine):
        import requests
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("403")
        with patch("insights.ai_insights.requests.post", return_value=mock_resp):
            text, err = engine._call_pollinations("prompt", "system")
        assert text is None
        assert len(err) > 0


# ---------------------------------------------------------------------------
# _call_ollamafree
# ---------------------------------------------------------------------------

class TestCallOllamaFree:
    def _mock_ollama(self, response_text):
        mock_client = MagicMock()
        mock_client.chat.return_value = response_text
        mock_module = MagicMock()
        mock_module.OllamaFreeAPI.return_value = mock_client
        return mock_module

    def test_success_returns_text(self, engine):
        mock_module = self._mock_ollama("Ollama answer")
        with patch.dict(sys.modules, {"ollamafreeapi": mock_module}):
            text, err = engine._call_ollamafree("prompt", "system")
        assert text == "Ollama answer"
        assert err == ""

    def test_strips_think_tags(self, engine):
        mock_module = self._mock_ollama("<think>internal reasoning</think>Final answer")
        with patch.dict(sys.modules, {"ollamafreeapi": mock_module}):
            text, err = engine._call_ollamafree("prompt", "system")
        assert text == "Final answer"
        assert "<think>" not in text

    def test_strips_multiline_think_tags(self, engine):
        raw = "<think>\nline 1\nline 2\n</think>\nActual output"
        mock_module = self._mock_ollama(raw)
        with patch.dict(sys.modules, {"ollamafreeapi": mock_module}):
            text, err = engine._call_ollamafree("prompt", "system")
        assert "Actual output" in text
        assert "<think>" not in text

    def test_only_think_content_returns_none(self, engine):
        mock_module = self._mock_ollama("<think>only reasoning</think>")
        with patch.dict(sys.modules, {"ollamafreeapi": mock_module}):
            text, err = engine._call_ollamafree("prompt", "system")
        assert text is None

    def test_empty_response_returns_none(self, engine):
        mock_module = self._mock_ollama("")
        with patch.dict(sys.modules, {"ollamafreeapi": mock_module}):
            text, err = engine._call_ollamafree("prompt", "system")
        assert text is None

    def test_import_error_returns_none(self, engine):
        with patch.dict(sys.modules, {"ollamafreeapi": None}):
            text, err = engine._call_ollamafree("prompt", "system")
        assert text is None
        assert len(err) > 0


# ---------------------------------------------------------------------------
# Fallback chain
# ---------------------------------------------------------------------------

class TestFallbackChain:
    def test_uses_pollinations_first(self, engine):
        with patch.object(engine, "_call_pollinations", return_value=("Pollinations wins", "")) as mock_p, \
             patch.object(engine, "_call_ollamafree", return_value=(None, "err")) as mock_o:
            result = engine._call_with_fallback("prompt", "system")
        assert result == "Pollinations wins"
        mock_p.assert_called_once()
        mock_o.assert_not_called()

    def test_falls_through_to_ollamafree_when_pollinations_fails(self, engine):
        with patch.object(engine, "_call_pollinations", return_value=(None, "timeout")), \
             patch.object(engine, "_call_ollamafree", return_value=("Ollama wins", "")):
            result = engine._call_with_fallback("prompt", "system")
        assert result == "Ollama wins"

    def test_returns_error_message_when_all_fail(self, engine):
        with patch.object(engine, "_call_pollinations", return_value=(None, "timeout")), \
             patch.object(engine, "_call_ollamafree", return_value=(None, "unavailable")):
            result = engine._call_with_fallback("prompt", "system")
        assert "unavailable" in result.lower() or "failed" in result.lower()

    def test_error_message_includes_last_error(self, engine):
        with patch.object(engine, "_call_pollinations", return_value=(None, "p_err")), \
             patch.object(engine, "_call_ollamafree", return_value=(None, "o_err")):
            result = engine._call_with_fallback("prompt", "system")
        assert "OllamaFree" in result or "o_err" in result


# ---------------------------------------------------------------------------
# Public interface methods
# ---------------------------------------------------------------------------

class TestPublicMethods:
    def test_generate_performance_insights_delegates_to_fallback(
        self, engine, metrics, feature_importance_df
    ):
        with patch.object(engine, "_call_with_fallback", return_value="insight text") as mock:
            result = engine.generate_performance_insights(metrics, feature_importance_df)
        assert result == "insight text"
        mock.assert_called_once()

    def test_generate_performance_insights_with_empty_fi(self, engine, metrics):
        with patch.object(engine, "_call_with_fallback", return_value="ok"):
            result = engine.generate_performance_insights(metrics, pd.DataFrame())
        assert result == "ok"

    def test_generate_scenario_recommendations_delegates_to_fallback(self, engine):
        scenarios = {"Price +10%": {"new_revenue": 50000}, "Baseline": {"new_revenue": 45000}}
        with patch.object(engine, "_call_with_fallback", return_value="rec text") as mock:
            result = engine.generate_scenario_recommendations(scenarios)
        assert result == "rec text"
        mock.assert_called_once()

    def test_generate_executive_summary_delegates_to_fallback(self, engine, metrics):
        with patch.object(engine, "_call_with_fallback", return_value="summary text") as mock:
            result = engine.generate_executive_summary({"metrics": metrics})
        assert result == "summary text"
        mock.assert_called_once()

    def test_generate_executive_summary_with_empty_dict(self, engine):
        with patch.object(engine, "_call_with_fallback", return_value="ok"):
            result = engine.generate_executive_summary({})
        assert result == "ok"
