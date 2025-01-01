import requests
import time
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ModerationResult:
    severity: float
    categories: Dict[str, bool]
    action: str
    message: str
    timeout_duration: Optional[int] = None

class ContentModerator:
    def __init__(self, config: dict):
        self.config = config
        self.moderation_config = config.get('moderation', {})
        self.api_config = self.moderation_config.get('api', {})
        self.settings = self.moderation_config.get('settings', {})
        
        # Initialize violation tracking
        self.user_violations = {}
        
    def check_content(self, content: str, user_id: str) -> ModerationResult:
        if not self.moderation_config.get('enabled', False):
            return ModerationResult(0, {}, "allow", "")
            
        # Get content analysis from Azure
        severity, categories = self._analyze_content(content)
        
        # Get appropriate action based on severity
        action, message, timeout = self._get_action(severity, user_id)
        
        return ModerationResult(
            severity=severity,
            categories=categories,
            action=action,
            message=message,
            timeout_duration=timeout
        )
    
    def _analyze_content(self, content: str) -> Tuple[float, Dict[str, bool]]:
        if not self.api_config.get('key') or not self.api_config.get('endpoint'):
            return 0, {}
            
        try:
            headers = {
                'Ocp-Apim-Subscription-Key': self.api_config['key'],
                'Content-Type': 'text/plain'
            }
            
            # Azure Content Moderator API call
            response = requests.post(
                f"{self.api_config['endpoint']}/contentmoderator/moderate/v1.0/ProcessText/Screen",
                headers=headers,
                data=content.encode('utf-8')
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Calculate severity based on API response
                severity = self._calculate_severity(result)
                
                # Extract categories
                categories = {
                    'profanity': bool(result.get('Terms')),
                    'sexual_content': result.get('Classification', {}).get('ReviewRecommended', False),
                    'hate_speech': False,  # Azure doesn't directly detect this
                    'violence': False      # Azure doesn't directly detect this
                }
                
                return severity, categories
                
        except Exception as e:
            print(f"Moderation API error: {e}")
            
        return 0, {}
    
    def _calculate_severity(self, api_result: dict) -> float:
        severity = 0
        
        # Check for profanity terms
        terms = api_result.get('Terms', [])
        if terms:
            severity = max(severity, 0.5 + (len(terms) * 0.1))
            
        # Check classification scores
        classification = api_result.get('Classification', {})
        if classification.get('ReviewRecommended'):
            severity = max(severity, 0.7)
            
        return min(severity, 1.0)
    
    def _get_action(self, severity: float, user_id: str) -> Tuple[str, str, Optional[int]]:
        auto_mod = self.settings.get('auto_moderation', {})
        severity_levels = self.settings.get('severity_levels', {})
        actions = self.settings.get('actions', {})
        
        # Update user violations
        if severity >= auto_mod.get('delete_threshold', 0.8):
            self.user_violations[user_id] = self.user_violations.get(user_id, 0) + 1
            
        # Check for timeout based on repeated violations
        if self.user_violations.get(user_id, 0) >= auto_mod.get('max_violations', 3):
            timeout_duration = auto_mod.get('timeout_duration', 300)
            return "timeout", actions['high']['message'], timeout_duration
            
        # Determine action based on severity
        if severity >= severity_levels.get('high', 0.8):
            return "delete", actions['high']['message'], actions['high']['duration']
        elif severity >= severity_levels.get('medium', 0.6):
            return "delete", actions['medium']['message'], None
        elif severity >= severity_levels.get('low', 0.3):
            return "warn", actions['low']['message'], None
            
        return "allow", "", None
    
    def reset_violations(self, user_id: str):
        """Reset violation count for a user"""
        if user_id in self.user_violations:
            del self.user_violations[user_id]
