import unittest

from streamlit.testing.v1 import AppTest


class AppSmokeTests(unittest.TestCase):
    def test_app_starts_without_foundry_secrets(self):
        app = AppTest.from_file("../app.py", default_timeout=30).run()
        self.assertFalse(app.exception)


if __name__ == "__main__":
    unittest.main()
