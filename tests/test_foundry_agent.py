import unittest
from unittest.mock import Mock, patch

from foundry_agent import FoundrySettings, ask_foundry, classify_question, select_evidence


class FoundryAgentTests(unittest.TestCase):
    def test_endpoint_normalization(self):
        settings = FoundrySettings(
            endpoint="https://example.openai.azure.com/openai/v1/",
            api_key="secret",
            deployment="model",
        )
        self.assertEqual(
            settings.chat_completions_url,
            "https://example.openai.azure.com/openai/v1/chat/completions",
        )

    def test_classifies_question(self):
        self.assertEqual(classify_question("哪一組側壁角度最高？"), "排序查詢")
        self.assertEqual(classify_question("Cl2 增加有什麼影響？"), "製程影響")

    def test_highest_angle_is_selected_first(self):
        records = [
            {"source": "P001", "case": "a", "angle": 80.0},
            {"source": "P002", "case": "b", "angle": 89.0},
        ]
        evidence = select_evidence(records, "哪一組側壁角度最高？", limit=2)
        self.assertEqual(evidence[0]["reference"], "P002")

    @patch("foundry_agent.requests.post")
    def test_foundry_request_uses_secret_without_exposing_it(self, post):
        response = Mock(ok=True)
        response.json.return_value = {
            "choices": [{"message": {"content": "依據 [P001]，目前資料不足。"}}]
        }
        post.return_value = response
        settings = FoundrySettings(
            endpoint="https://example.openai.azure.com",
            api_key="not-logged-secret",
            deployment="gpt-deployment",
        )

        answer = ask_foundry(
            settings,
            "Cl2 的影響？",
            [{"reference": "P001"}],
            {"Cl2_sccm": 40},
        )

        self.assertIn("[P001]", answer)
        request = post.call_args
        self.assertEqual(request.kwargs["headers"]["api-key"], "not-logged-secret")
        self.assertEqual(request.kwargs["json"]["model"], "gpt-deployment")


if __name__ == "__main__":
    unittest.main()
