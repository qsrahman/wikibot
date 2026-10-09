import os
import unittest
from unittest import mock

from wikibot.config import Config


def _env(**values):
    return mock.patch.dict(os.environ, values, clear=True)


class ConfigFromEnv(unittest.TestCase):
    def test_reads_and_strips_values(self):
        with _env(OPENAI_API_KEY=" key ", OPENAI_MODEL=" m ", OPENAI_BASE_URL=" http://x/v1 "):
            cfg = Config.from_env()
        self.assertEqual((cfg.api_key, cfg.model, cfg.base_url), ("key", "m", "http://x/v1"))

    def test_blank_base_url_is_absent(self):
        with _env(OPENAI_API_KEY="key", OPENAI_MODEL="m", OPENAI_BASE_URL="  "):
            self.assertIsNone(Config.from_env().base_url)

    def test_missing_or_blank_required_var_is_named(self):
        cases = {
            "OPENAI_API_KEY": {"OPENAI_MODEL": "m"},
            "OPENAI_MODEL": {"OPENAI_API_KEY": "key", "OPENAI_MODEL": "   "},
        }
        for name, env in cases.items():
            with self.subTest(name), _env(**env), self.assertRaises(SystemExit) as raised:
                Config.from_env()
            # This message is the startup contract in AGENTS.md.
            self.assertEqual(
                str(raised.exception),
                f"error: {name} is not set — fill it in .env (see README.md)",
            )
