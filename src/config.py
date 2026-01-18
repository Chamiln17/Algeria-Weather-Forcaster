"""
Configuration Management Module
Loads settings from config.yaml and provides easy access.
"""

import yaml
from pathlib import Path
from typing import Any, Dict

class Config:
    """Project configuration manager"""
    
    def __init__(self, config_path: str = "../config.yaml"):
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Example:
            config.get('paths.raw_data')
            config.get('lstm.look_back')
        """
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    @property
    def paths(self) -> Dict[str, str]:
        """Get all path configurations"""
        return self._config.get('paths', {})
    
    @property
    def preprocessing(self) -> Dict[str, Any]:
        """Get preprocessing configurations"""
        return self._config.get('preprocessing', {})
    
    @property
    def forecasting(self) -> Dict[str, Any]:
        """Get forecasting configurations"""
        return self._config.get('forecasting', {})
    
    def __repr__(self):
        return f"Config(path='{self.config_path}')"


# Global config instance
_config = None

def get_config() -> Config:
    """Get or create global config instance"""
    global _config
    if _config is None:
        _config = Config()
    return _config


# Convenience functions
def get_path(key: str) -> Path:
    """Get path from config and return as Path object"""
    return Path(get_config().get(f'paths.{key}'))

def get_random_seed(library: str = 'numpy') -> int:
    """Get random seed for specified library"""
    return get_config().get(f'random_seeds.{library}', 42)
