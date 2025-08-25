import os
import json
import requests
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)

class OpenAIAnalyzer:
    def __init__(self):
        """Initialize OpenAI client with Azure OpenAI configuration"""
        self.api_key = "1VktmuHQa8cP22aRbk03Aj6j5DSKfpIBgXCzbncvT3Rz6AAUJQQJ99BHACPV0roXJ3w3AAABACOGo2cK"
        self.endpoint = "https://openai-dev-aiops.openai.azure.com/openai/deployments/AIOPS_DEV_OAI_gpt-4o/chat/completions?api-version=2025-01-01-preview"
        self.deployment_name = "AIOPS_DEV_OAI_gpt-4o"
        self.model = "gpt-4o"
        
        # FIXED: Headers for Azure OpenAI API - use api-key instead of Authorization Bearer
        self.headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key  # Changed from "Authorization": f"Bearer {self.api_key}"
        }
        
        logger.info("OpenAI Analyzer initialized with Azure OpenAI configuration")
    
    def _make_api_request(self, messages: List[Dict], max_tokens: int = 1000, temperature: float = 0.3) -> Optional[str]:
        """Make API request to Azure OpenAI"""
        try:
            payload = {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.95,
                "frequency_penalty": 0,
                "presence_penalty": 0
            }
            
            response = requests.post(
                self.endpoint,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                logger.error(f"API request failed with status {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error("API request timed out")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in API request: {str(e)}")
            return None
    
    def classify_incident(self, description: str, service_offering: str, assignment_group: str) -> Dict[str, Any]:
        """Classify an incident using OpenAI GPT-4"""
        categories = [
            "Application", "Network", "Hardware", "Database", "Security", 
            "Performance", "Access Management", "Storage", "Backup & Recovery",
            "Email & Communication", "Infrastructure", "Software", "Other"
        ]
        
        messages = [
            {
                "role": "system",
                "content": """You are an expert IT incident classifier. Analyze the provided incident details and classify it into one of the predefined categories. 
                
                Categories: Application, Network, Hardware, Database, Security, Performance, Access Management, Storage, Backup & Recovery, Email & Communication, Infrastructure, Software, Other
                
                Respond with ONLY the category name, nothing else."""
            },
            {
                "role": "user", 
                "content": f"""Classify this incident:
                
Description: {description}
Service Offering: {service_offering}
Assignment Group: {assignment_group}

Category:"""
            }
        ]
        
        response = self._make_api_request(messages, max_tokens=50, temperature=0.1)
        
        if response and response.strip() in categories:
            return {
                "category": response.strip(),
                "confidence": 0.85,  # Default confidence for LLM classification
                "method": "LLM"
            }
        else:
            logger.warning(f"Invalid or no response from OpenAI: {response}")
            return {
                "category": "Other",
                "confidence": 0.5,
                "method": "LLM"
            }
    
    def analyze_tcd_data(self, changes: List[Dict]) -> str:
        """Analyze Technical Change Decision data using OpenAI"""
        changes_text = json.dumps(changes, indent=2)
        
        messages = [
            {
                "role": "system",
                "content": """You are an expert IT change management analyst. Analyze the provided change data and provide insights on risk assessment, implementation recommendations, and potential impact on incidents."""
            },
            {
                "role": "user",
                "content": f"""Analyze the following changes and provide recommendations:

{changes_text}

Please provide:
1. Risk assessment summary
2. Implementation recommendations
3. Potential incident impact
4. Best practices to follow"""
            }
        ]
        
        response = self._make_api_request(messages, max_tokens=800)
        return response or "Analysis unavailable at this time."
    
    def analyze_recurring_incidents(self, incidents_data: List[Dict], batch_size: int = 64) -> Dict[str, Any]:
        """Analyze recurring incidents using OpenAI"""
        # Process in batches to avoid token limits
        results = {
            "clusters": [],
            "summary": "",
            "recommendations": []
        }
        
        try:
            # Sample data for analysis if too large
            sample_size = min(len(incidents_data), 50)
            sampled_incidents = incidents_data[:sample_size]
            
            incidents_text = ""
            for i, incident in enumerate(sampled_incidents):
                incidents_text += f"Incident {i+1}:\n"
                incidents_text += f"Description: {incident.get('short_description', 'N/A')}\n"
                incidents_text += f"Category: {incident.get('category', 'N/A')}\n"
                incidents_text += f"Assignment Group: {incident.get('assignment_group', 'N/A')}\n"
                incidents_text += f"Priority: {incident.get('priority', 'N/A')}\n"
                incidents_text += "---\n"
            
            messages = [
                {
                    "role": "system",
                    "content": """You are an expert IT incident analyst. Analyze the provided incidents to identify recurring patterns, common root causes, and provide actionable recommendations.
                    
                    Please identify clusters of similar incidents and provide:
                    1. Summary of each cluster
                    2. Common patterns
                    3. Recommended actions
                    4. Root cause analysis"""
                },
                {
                    "role": "user",
                    "content": f"""Analyze these incidents for recurring patterns:

{incidents_text}

Please provide your analysis in a structured format identifying clusters and recommendations."""
                }
            ]
            
            response = self._make_api_request(messages, max_tokens=1200)
            
            if response:
                # Parse response into structured format
                results["summary"] = response
                
                # Create mock clusters based on analysis
                # In a real implementation, you'd parse the AI response more sophisticatedly
                results["clusters"] = [
                    {
                        "summary": "Application Performance Issues",
                        "occurrences": len([i for i in sampled_incidents if 'performance' in str(i.get('short_description', '')).lower()]),
                        "avg_mttr": 4.5,
                        "incidents": [i for i in sampled_incidents if 'performance' in str(i.get('short_description', '')).lower()][:5]
                    },
                    {
                        "summary": "Network Connectivity Problems", 
                        "occurrences": len([i for i in sampled_incidents if any(word in str(i.get('short_description', '')).lower() for word in ['network', 'connection', 'connectivity'])]),
                        "avg_mttr": 3.2,
                        "incidents": [i for i in sampled_incidents if any(word in str(i.get('short_description', '')).lower() for word in ['network', 'connection', 'connectivity'])][:5]
                    }
                ]
                
        except Exception as e:
            logger.error(f"Error in recurring incidents analysis: {str(e)}")
            results["summary"] = "Analysis could not be completed due to an error."
            
        return results
    
    def generate_rca_for_clusters(self, clusters: List[Dict]) -> List[Dict]:
        """Generate Root Cause Analysis for incident clusters"""
        rca_results = []
        
        for cluster in clusters:
            cluster_summary = cluster.get('summary', 'Unknown cluster')
            incidents = cluster.get('incidents', [])
            
            # Create incident summary for analysis
            incident_details = ""
            for incident in incidents[:5]:  # Limit to 5 incidents per cluster
                incident_details += f"- {incident.get('short_description', 'N/A')}\n"
            
            messages = [
                {
                    "role": "system",
                    "content": """You are an expert IT incident analyst specializing in root cause analysis. Provide detailed RCA and actionable recommendations for the given incident cluster."""
                },
                {
                    "role": "user",
                    "content": f"""Perform root cause analysis for this incident cluster:

Cluster: {cluster_summary}
Occurrences: {cluster.get('occurrences', 0)}
Sample Incidents:
{incident_details}

Please provide:
1. Detailed root cause analysis
2. Specific actionable recommendations
3. Prevention strategies"""
                }
            ]
            
            response = self._make_api_request(messages, max_tokens=600)
            
            if response:
                # Split response into RCA and recommendations
                parts = response.split("Recommendations:")
                rca = parts[0].replace("Root Cause Analysis:", "").strip()
                recommendations = parts[1].strip() if len(parts) > 1 else "No specific recommendations provided."
                
                rca_results.append({
                    "cluster": cluster_summary,
                    "rca": rca,
                    "recommendations": recommendations
                })
            else:
                rca_results.append({
                    "cluster": cluster_summary,
                    "rca": "RCA could not be generated at this time.",
                    "recommendations": "No recommendations available."
                })
        
        return rca_results
    
    def get_reopen_insights(self, prompt_text: str) -> str:
        """Analyze reopened incidents and provide insights"""
        messages = [
            {
                "role": "system",
                "content": """You are an expert IT service management analyst. Analyze the provided reopened incidents data and provide actionable insights on why incidents are being reopened and how to prevent it."""
            },
            {
                "role": "user",
                "content": prompt_text
            }
        ]
        
        response = self._make_api_request(messages, max_tokens=800)
        return response or "Insights could not be generated at this time."
    
    def analyze_sentiment(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze sentiment of incident descriptions"""
        # For basic sentiment analysis, we can use OpenAI
        # But for performance, you might want to use a dedicated sentiment analysis library
        
        sample_texts = texts[:10]  # Analyze sample for performance
        texts_combined = "\n".join([f"{i+1}. {text}" for i, text in enumerate(sample_texts)])
        
        messages = [
            {
                "role": "system",
                "content": """Analyze the sentiment of the provided incident descriptions. Classify each as Positive, Negative, or Neutral and provide an overall sentiment score from -1 (very negative) to 1 (very positive)."""
            },
            {
                "role": "user", 
                "content": f"""Analyze sentiment for these incident descriptions:

{texts_combined}

Provide overall sentiment analysis with percentages."""
            }
        ]
        
        response = self._make_api_request(messages, max_tokens=400)
        
        # Return mock sentiment data since real analysis would require more sophisticated parsing
        return {
            "overall_sentiment": "Neutral",
            "positive_percentage": 25.0,
            "negative_percentage": 45.0, 
            "neutral_percentage": 30.0,
            "average_polarity": -0.2,
            "analysis": response or "Sentiment analysis unavailable"
        }
    
    def health_check(self) -> bool:
        """Check if OpenAI API is accessible"""
        try:
            messages = [
                {
                    "role": "user",
                    "content": "Hello, please respond with 'OK' to confirm you're working."
                }
            ]
            
            response = self._make_api_request(messages, max_tokens=10, temperature=0)
            return response is not None and "OK" in response.upper()
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False
