# Add these methods to your DashboardGenerator class (src/dashboard_DD.py)

import pandas as pd
from src.openai_client import OpenAIAnalyzer

class DashboardGenerator:
    def __init__(self):
        self.openai_analyzer = OpenAIAnalyzer()
        # ... existing initialization code ...
    
    def get_openai_reopen_insights(self, prompt_text: str) -> str:
        """Get insights on reopened incidents using OpenAI"""
        try:
            return self.openai_analyzer.get_reopen_insights(prompt_text)
        except Exception as e:
            print(f"Error getting OpenAI insights: {str(e)}")
            return "Unable to generate insights at this time. Please check OpenAI configuration."
    
    def format_reopen_prompt(self, df_reopened: pd.DataFrame) -> str:
        """Format prompt for OpenAI analysis of reopened incidents"""
        # Limit to top 20 reopened incidents for analysis
        sample_size = min(20, len(df_reopened))
        df_sample = df_reopened.head(sample_size)
        
        prompt = f"""Analyze the following {sample_size} reopened incidents and provide insights:

REOPENED INCIDENTS DATA:
"""
        
        for idx, row in df_sample.iterrows():
            prompt += f"""
Incident {idx + 1}:
- Number: {row.get('Number', 'N/A')}
- Category: {row.get('Predicted Category', 'N/A')}
- Description: {row.get('Short description', 'N/A')[:150]}...
- Assignment Group: {row.get('Assignment group', 'N/A')}
- Priority: {row.get('Priority', 'N/A')}
- Service Offering: {row.get('Service offering', 'N/A')}
- Reopen Count: {row.get('Reopen count', 'N/A')}
"""
        
        prompt += """

Please provide:
1. Common patterns in these reopened incidents
2. Likely root causes for reopening
3. Specific recommendations to reduce reopen rates
4. Process improvements that could be implemented
5. Any notable trends by category or assignment group

Focus on actionable insights that can help prevent future reopenings."""
        
        return prompt
    
    def analyze_recurring_incidents(self, df: pd.DataFrame, eps: float = 0.3, batch_size: int = 64) -> dict:
        """Analyze recurring incidents using OpenAI"""
        try:
            # Convert DataFrame to list of dictionaries
            incidents_data = df.to_dict('records')
            
            # Use OpenAI analyzer for recurring incidents analysis
            result = self.openai_analyzer.analyze_recurring_incidents(incidents_data, batch_size)
            
            return result
            
        except Exception as e:
            print(f"Error in recurring incidents analysis: {str(e)}")
            return {
                "clusters": [],
                "summary": "Analysis could not be completed due to an error.",
                "recommendations": []
            }
    
    def generate_rca_for_clusters(self, clusters: list) -> list:
        """Generate Root Cause Analysis for clusters using OpenAI"""
        try:
            return self.openai_analyzer.generate_rca_for_clusters(clusters)
        except Exception as e:
            print(f"Error generating RCA: {str(e)}")
            return [{"cluster": "Error", "rca": "Could not generate RCA", "recommendations": "N/A"}]
    
    def openai_health_check(self) -> bool:
        """Check if OpenAI service is available"""
        try:
            return self.openai_analyzer.health_check()
        except Exception:
            return False
