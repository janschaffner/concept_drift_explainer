import openai
import json
import time
from typing import Dict, List, Any
from config.settings import OPENAI_API_KEY

class LLMAnalyzer:
    """
    Integration with GPT-4o for context-aware analysis of change points
    """
    
    def __init__(self, api_key=None, model="gpt-4o", temperature=0.0):
        """
        Initialize the LLM analyzer
        
        Parameters:
        -----------
        api_key : str, optional
            OpenAI API key (defaults to env variable if not provided)
        model : str
            Model to use (default: gpt-4o)
        temperature : float
            Temperature parameter for the LLM (0.0-1.0)
        """
        # Use the provided API key or fall back to the environment variable
        self.api_key = api_key or OPENAI_API_KEY
        
        if not self.api_key:
            raise ValueError("No OpenAI API key provided. Set OPENAI_API_KEY in .env or pass as parameter.")
            
        openai.api_key = self.api_key
        self.model = model
        self.temperature = temperature
    
    def create_context_prompt(self, 
                             change_point: Dict[str, Any], 
                             internal_context: List[Dict[str, Any]], 
                             external_context: List[Dict[str, Any]],
                             event_log_sample: Dict[str, Any] = None) -> str:
        """
        Create a prompt for the LLM to analyze the change point with context
        
        Parameters:
        -----------
        change_point : Dict
            Change point details
        internal_context : List[Dict]
            Internal context events around the change point timeframe
        external_context : List[Dict]
            External context events around the change point timeframe
        event_log_sample : Dict, optional
            Sample of the event log before and after the change point
            
        Returns:
        --------
        str
            Formatted prompt for the LLM
        """
        
        # Format the timestamp for better readability
        timestamp = change_point.get("timestamp", "")
        if timestamp:
            try:
                # Format date if it's a datetime object or string
                if hasattr(timestamp, "strftime"):
                    formatted_date = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    # Assuming ISO format string
                    from datetime import datetime
                    formatted_date = datetime.fromisoformat(timestamp).strftime("%Y-%m-%d %H:%M:%S")
            except:
                formatted_date = timestamp
        else:
            formatted_date = "Unknown date"
        
        # Build the prompt
        prompt = f"""You are an expert process mining analyst with deep knowledge of business processes and organizational change management. Your task is to analyze a detected change point in a process and explain its potential causes based on contextual information.

CHANGE POINT DETAILS:
- Timestamp: {formatted_date}
- Process ID: {change_point.get('process_id', 'Unknown')}
- Change Type: {change_point.get('change_type', 'Unknown')}
- Confidence: {change_point.get('confidence', 'Unknown')}
- Affected Attributes: {change_point.get('affected_attributes', [])}

INTERNAL CONTEXT EVENTS (organizational changes, system updates, etc.):
"""
        
        if internal_context:
            for idx, event in enumerate(internal_context, 1):
                prompt += f"{idx}. {event.get('date', 'Unknown date')}: {event.get('description', 'No description')}\n"
                if event.get('details'):
                    prompt += f"   Details: {event['details']}\n"
                prompt += f"   Source: {event.get('source', 'Unknown source')}\n\n"
        else:
            prompt += "No internal context events found in the specified timeframe.\n\n"
            
        prompt += "EXTERNAL CONTEXT EVENTS (regulations, market changes, etc.):\n"
        
        if external_context:
            for idx, event in enumerate(external_context, 1):
                prompt += f"{idx}. {event.get('date', 'Unknown date')}: {event.get('description', 'No description')}\n"
                if event.get('details'):
                    prompt += f"   Details: {event['details']}\n"
                prompt += f"   Source: {event.get('source', 'Unknown source')}\n\n"
        else:
            prompt += "No external context events found in the specified timeframe.\n\n"
        
        if event_log_sample:
            prompt += "EVENT LOG SAMPLE:\n"
            prompt += f"Before change point: {event_log_sample.get('before', 'No data')}\n\n"
            prompt += f"After change point: {event_log_sample.get('after', 'No data')}\n\n"
        
        prompt += """ANALYSIS TASK:
1. Analyze the potential relationship between the context events and the detected change point.
2. Identify which context events (if any) are likely to have caused or influenced the process change.
3. Explain your reasoning and the potential mechanisms of influence.
4. Rate your confidence in the causal relationship (Low/Medium/High) and explain why.
5. If multiple factors could have contributed, rank them by likely importance.

YOUR ANALYSIS:"""

        return prompt
    
    def analyze_change_point(self, 
                            change_point: Dict[str, Any], 
                            internal_context: List[Dict[str, Any]], 
                            external_context: List[Dict[str, Any]],
                            event_log_sample: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Use the LLM to analyze a change point with context
        
        Parameters as in create_context_prompt()
            
        Returns:
        --------
        Dict
            LLM analysis results
        """
        
        # Create the prompt
        prompt = self.create_context_prompt(
            change_point, 
            internal_context, 
            external_context,
            event_log_sample
        )
        
        # Call the LLM API
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert process mining analyst specializing in concept drift analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature
            )
            
            analysis_text = response.choices[0].message.content
            
            # Attempt to extract structured information from the analysis
            # Note: This is a simplistic approach; a more robust parsing could be implemented
            
            # Extract confidence rating
            confidence_level = "Unknown"
            if "confidence: high" in analysis_text.lower():
                confidence_level = "High"
            elif "confidence: medium" in analysis_text.lower():
                confidence_level = "Medium"
            elif "confidence: low" in analysis_text.lower():
                confidence_level = "Low"
            
            # Return the analysis results
            return {
                "change_point_id": change_point.get("id", ""),
                "change_point_timestamp": change_point.get("timestamp", ""),
                "analysis_text": analysis_text,
                "confidence_level": confidence_level,
                "prompt_used": prompt,
                "model_used": self.model,
                "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            # Handle API errors
            return {
                "change_point_id": change_point.get("id", ""),
                "error": str(e),
                "prompt_used": prompt,
                "model_used": self.model,
                "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
