# Test script - save as test_openai.py
from src.openai_client import OpenAIAnalyzer

analyzer = OpenAIAnalyzer()
result = analyzer.health_check()
print(f"Connection successful: {result}")
