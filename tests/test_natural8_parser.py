# tests/test_natural8_parser.py
import pytest
from src.parsers.natural8_parser import Natural8Parser

def test_basic_parsing():
    parser = Natural8Parser()
    # Basic test - will work with mock data
    hands = parser.parse_file("nonexistent.txt")
    assert hands == []