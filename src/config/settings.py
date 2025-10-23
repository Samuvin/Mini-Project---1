"""
Application configuration using dependency injection principles.
Centralized configuration management (Single Responsibility).
"""

from dataclasses import dataclass, field
from typing import Dict, Any
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for model loading."""
    
    models_directory: str = 'models'
    available_modalities: list = field(default_factory=lambda: ['speech', 'handwriting', 'gait'])
    cache_models: bool = True


@dataclass
class CalibrationConfig:
    """Configuration for calibration service."""
    
    strategy: str = 'feature_aware'
    temperature: float = 1.0


@dataclass
class EnsembleConfig:
    """Configuration for ensemble predictions."""
    
    default_voting_method: str = 'soft'
    require_min_modalities: int = 1


@dataclass
class ApplicationConfig:
    """Main application configuration."""
    
    model_config: ModelConfig = field(default_factory=ModelConfig)
    calibration_config: CalibrationConfig = field(default_factory=CalibrationConfig)
    ensemble_config: EnsembleConfig = field(default_factory=EnsembleConfig)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ApplicationConfig':
        """Create configuration from dictionary."""
        return cls(
            model_config=ModelConfig(**config_dict.get('model', {})),
            calibration_config=CalibrationConfig(**config_dict.get('calibration', {})),
            ensemble_config=EnsembleConfig(**config_dict.get('ensemble', {}))
        )
    
    @classmethod
    def default(cls) -> 'ApplicationConfig':
        """Get default configuration."""
        return cls()


class ConfigurationManager:
    """Manages application configuration (Singleton pattern)."""
    
    _instance: 'ConfigurationManager' = None
    _config: ApplicationConfig = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def initialize(self, config: ApplicationConfig = None):
        """Initialize configuration."""
        self._config = config or ApplicationConfig.default()
    
    def get_config(self) -> ApplicationConfig:
        """Get current configuration."""
        if self._config is None:
            self.initialize()
        return self._config
    
    @property
    def model(self) -> ModelConfig:
        """Get model configuration."""
        return self.get_config().model_config
    
    @property
    def calibration(self) -> CalibrationConfig:
        """Get calibration configuration."""
        return self.get_config().calibration_config
    
    @property
    def ensemble(self) -> EnsembleConfig:
        """Get ensemble configuration."""
        return self.get_config().ensemble_config


def get_config_manager() -> ConfigurationManager:
    """Get configuration manager instance."""
    return ConfigurationManager()

