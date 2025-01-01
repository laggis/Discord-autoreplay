import re
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import json
import os
import yaml

@dataclass
class ModerationResult:
    severity: float
    categories: Dict[str, bool]
    action: str
    message: str
    timeout_duration: Optional[int] = None

class ContentModerator:
    def __init__(self, config: dict):
        self.config_path = 'config.yaml'  # Store config path for reloading
        self.update_config(config)
        print("Local content moderator initialized")
    
    def update_config(self, config: dict):
        """Update moderator with new config"""
        self.config = config
        self.moderation_config = config.get('moderation', {})
        self.settings = self.moderation_config.get('settings', {})
        
        # Initialize violation tracking
        self.user_violations = {}
        
        self.load_word_lists()
        print("Moderator config updated")
    
    def reload_config(self):
        """Reload config from file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                self.update_config(config)
                return True
        except Exception as e:
            print(f"Error reloading config: {e}")
            return False
    
    def load_word_lists(self):
        """Load word lists from files or create default ones"""
        self.word_lists = {
            'profanity': set([
                'fuck', 'shit', 'ass', 'bitch', 'dick', 'pussy', 'cock', 'cunt',
                'bastard', 'motherfucker', 'whore', 'slut'
            ]),
            'hate_speech': set([
                'nigger', 'faggot', 'retard', 'spic', 'kike', 'chink', 'wetback',
                'nazi', 'hitler'
            ]),
            'mild_profanity': set([
                'damn', 'hell', 'crap', 'piss', 'suck', 'butt', 'prick',
                'wtf', 'stfu', 'fck', 'fuk', 'fck'
            ])
        }
        
        # Add custom words from config
        custom_words = self.config.get('moderation', {}).get('swear_words', [])
        self.word_lists['custom'] = set(custom_words)
    
    def check_content(self, content: str, user_id: str) -> ModerationResult:
        if not self.moderation_config.get('enabled', False):
            return ModerationResult(0, {}, "allow", "")
            
        # Check content against word lists
        severity, categories = self._analyze_content(content)
        
        # Get appropriate action based on severity
        action, message, timeout = self._get_action(severity, user_id)
        
        # Log the action if it's not "allow"
        if action != "allow":
            from web_config import log_moderation_action
            log_moderation_action(user_id, action, severity, categories)
        
        # Print moderation result for debugging
        print(f"Moderation result for message '{content}': severity={severity}, action={action}")
        
        return ModerationResult(
            severity=severity,
            categories=categories,
            action=action,
            message=message,
            timeout_duration=timeout
        )
    
    def _analyze_content(self, content: str) -> Tuple[float, Dict[str, bool]]:
        content_lower = content.lower()
        words = set(re.findall(r'\b\w+\b', content_lower))
        
        # Initialize categories
        categories = {
            'profanity': False,
            'hate_speech': False,
            'mild_profanity': False,
            'custom': False
        }
        
        # Check each category
        severity = 0
        for category, word_list in self.word_lists.items():
            matches = words.intersection(word_list)
            if matches:
                categories[category] = True
                # Calculate severity based on category and number of matches
                if category == 'hate_speech':
                    severity = max(severity, 0.9)  # Highest severity
                elif category == 'profanity':
                    severity = max(severity, 0.7 + (len(matches) * 0.1))
                elif category == 'mild_profanity':
                    severity = max(severity, 0.3 + (len(matches) * 0.1))
                elif category == 'custom':
                    severity = max(severity, 0.5 + (len(matches) * 0.1))
        
        # Check for letter repetition (e.g., "fuuuuck")
        for category, word_list in self.word_lists.items():
            for word in word_list:
                pattern = ''.join([f'{c}+' for c in word])
                if re.search(pattern, content_lower):
                    categories[category] = True
                    severity = max(severity, 0.7)
        
        return min(severity, 1.0), categories
    
    def _get_action(self, severity: float, user_id: str) -> Tuple[str, str, Optional[int]]:
        auto_mod = self.settings.get('auto_moderation', {})
        severity_levels = self.settings.get('severity_levels', {})
        actions = self.settings.get('actions', {})
        
        # Update user violations
        if severity >= auto_mod.get('delete_threshold', 0.7):
            self.user_violations[user_id] = self.user_violations.get(user_id, 0) + 1
            print(f"User {user_id} violations: {self.user_violations[user_id]}")
            
        # Check for timeout based on repeated violations
        if self.user_violations.get(user_id, 0) >= auto_mod.get('max_violations', 3):
            timeout_duration = auto_mod.get('timeout_duration', 300)
            return "timeout", actions['high']['message'], timeout_duration
            
        # Determine action based on severity
        if severity >= severity_levels.get('high', 0.8):
            return "delete", actions['high']['message'], actions['high'].get('duration')
        elif severity >= severity_levels.get('medium', 0.6):
            return "delete", actions['medium']['message'], None
        elif severity >= severity_levels.get('low', 0.3):
            return "warn", actions['low']['message'], None
            
        return "allow", "", None
    
    def reset_violations(self, user_id: str):
        """Reset violation count for a user"""
        if user_id in self.user_violations:
            del self.user_violations[user_id]
