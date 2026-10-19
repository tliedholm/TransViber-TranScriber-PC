"""Reading and writing track offsets: what the Offset column accepts and shows."""

from __future__ import annotations

import unittest

import tests  # noqa: F401  (puts app/ on sys.path)

from align import format_offset, parse_offset


class ParseOffset(unittest.TestCase):
    def test_accepted_forms(self):
        cases = {
            "150": 150.0,
            "150.25": 150.25,
            "2:30": 150.0,
            "2:30.5": 150.5,
            "2:30,5": 150.5,          # decimal comma
            "1:02:30": 3750.0,
            "1:02:30.250": 3750.25,
            "90:00": 5400.0,          # the first field can be any size
            "30:00:00": 108000.0,     # hours are not limited to 24
            "0": 0.0,
            "  2:30  ": 150.0,
            "0:00": 0.0,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(parse_offset(text), expected)

    def test_rounds_to_milliseconds(self):
        self.assertEqual(parse_offset("1.23456"), 1.235)

    def test_rejected_forms_say_why(self):
        cases = {
            "": "0 to reset",
            "   ": "0 to reset",
            "-5": "negative",
            "-0:30": "negative",
            "abc": "isn't a time",
            "nan": "isn't a time",
            "inf": "isn't a time",
            "1:75": "under 60",
            "1:60": "under 60",
            "1:02:60": "under 60",
            "1:2:3:4": "at most",
            "1.5:30": "isn't a time",  # only the last field takes a fraction
            "2:": "isn't a time",
            ":30": "isn't a time",
            "2:30.": "isn't a time",
            "1e3": "isn't a time",
            "+30": "isn't a time",
            "\u0661\u0665\u0660": "isn't a time",  # non-ASCII digits
            "9" * 400: "too large",
            "9" * 400 + ":00": "too large",
            "1" + "0" * 309: "too large",
            "999999999": "too large",   # over 100 days
        }
        for text, reason in cases.items():
            with self.subTest(text=text):
                with self.assertRaises(ValueError) as caught:
                    parse_offset(text)
                self.assertIn(reason, str(caught.exception))


class FormatOffset(unittest.TestCase):
    def test_standard_form(self):
        cases = {
            0: "0:00",
            5: "0:05",
            150: "2:30",
            150.5: "2:30.5",
            312.4: "5:12.4",
            150.25: "2:30.25",
            150.125: "2:30.125",
            3750: "1:02:30",
            108000: "30:00:00",       # no 24-hour wrap
            59.9999: "1:00",          # rounds to milliseconds first
        }
        for seconds, expected in cases.items():
            with self.subTest(seconds=seconds):
                self.assertEqual(format_offset(seconds), expected)

    def test_round_trips(self):
        for seconds in (0.0, 0.001, 1.5, 59.999, 61.0, 150.25, 3599.5, 3750.125, 90061.0):
            with self.subTest(seconds=seconds):
                self.assertEqual(parse_offset(format_offset(seconds)), seconds)


if __name__ == "__main__":
    unittest.main()
