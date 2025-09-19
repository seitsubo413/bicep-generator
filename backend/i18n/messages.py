import json
import os
from typing import Dict, Any, Optional

class MessageManager:
    """Multi-language message manager for backend"""
    
    def __init__(self, default_language: str = "ja"):
        self.default_language = default_language
        self.current_language = default_language
        self.messages: Dict[str, Dict[str, Any]] = {}
        self._load_messages()
    
    def _load_messages(self):
        """Load all language message files"""
        current_dir = os.path.dirname(__file__)
        
        # Load Japanese messages
        ja_path = os.path.join(current_dir, "locales", "ja.json")
        if os.path.exists(ja_path):
            with open(ja_path, "r", encoding="utf-8") as f:
                self.messages["ja"] = json.load(f)
        
        # Load English messages
        en_path = os.path.join(current_dir, "locales", "en.json")
        if os.path.exists(en_path):
            with open(en_path, "r", encoding="utf-8") as f:
                self.messages["en"] = json.load(f)
    
    def set_language(self, language: str):
        """Set current language"""
        if language in self.messages:
            self.current_language = language
        else:
            self.current_language = self.default_language
    
    def get(self, key: str, language: Optional[str] = None, **kwargs) -> str:
        """Get translated message by key"""
        lang = language or self.current_language
        
        # Try to get message from specified language
        if lang in self.messages:
            message = self._get_nested_value(self.messages[lang], key)
            if message:
                return self._format_message(message, **kwargs)
        
        # Fallback to default language
        if self.default_language in self.messages:
            message = self._get_nested_value(self.messages[self.default_language], key)
            if message:
                return self._format_message(message, **kwargs)
        
        # Return key if not found
        return f"[{key}]"
    
    def _get_nested_value(self, data: Dict[str, Any], key: str) -> Optional[str]:
        """Get value from nested dictionary using dot notation"""
        keys = key.split(".")
        current = data
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return None
        
        return current if isinstance(current, str) else None
    
    def _format_message(self, message: str, **kwargs) -> str:
        """Format message with parameters"""
        try:
            return message.format(**kwargs)
        except (KeyError, ValueError):
            return message

# Global message manager instance
message_manager = MessageManager()

def get_message(key: str, language: Optional[str] = None, **kwargs) -> str:
    """Convenience function to get translated message"""
    return message_manager.get(key, language, **kwargs)

def set_language(language: str):
    """Convenience function to set language"""
    message_manager.set_language(language)