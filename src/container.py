"""
Dependency Injection Container (DI Container pattern).
Implements Inversion of Control (IoC) for better testability and flexibility.
"""

from src.core.interfaces import IPredictionService, IModelLoader, ICalibrator
from src.services.model_loader_service import FileSystemModelLoader
from src.services.calibration_service import CalibrationServiceFactory
from src.services.prediction_service import MultiModalPredictionService
from src.config.settings import get_config_manager, ApplicationConfig


class ServiceContainer:
    """
    IoC container for dependency management.
    
    Implements Service Locator pattern with lazy initialization.
    All services are created through this container to ensure
    proper dependency injection.
    """
    
    def __init__(self, config: ApplicationConfig = None):
        """
        Initialize container with configuration.
        
        Args:
            config: Application configuration (optional)
        """
        self._config = config or get_config_manager().get_config()
        self._services = {}
    
    def get_model_loader(self) -> IModelLoader:
        """
        Get model loader service (singleton within container).
        
        Returns:
            Model loader instance
        """
        if 'model_loader' not in self._services:
            self._services['model_loader'] = FileSystemModelLoader(
                models_directory=self._config.model_config.models_directory
            )
        return self._services['model_loader']
    
    def get_calibrator(self) -> ICalibrator:
        """
        Get calibration service (singleton within container).
        
        Returns:
            Calibrator instance
        """
        if 'calibrator' not in self._services:
            self._services['calibrator'] = CalibrationServiceFactory.create(
                strategy_name=self._config.calibration_config.strategy
            )
        return self._services['calibrator']
    
    def get_prediction_service(self) -> IPredictionService:
        """
        Get prediction service (singleton within container).
        
        Dependencies are automatically resolved and injected.
        
        Returns:
            Prediction service instance
        """
        if 'prediction_service' not in self._services:
            model_loader = self.get_model_loader()
            calibrator = self.get_calibrator()
            
            self._services['prediction_service'] = MultiModalPredictionService(
                model_loader=model_loader,
                calibrator=calibrator
            )
        return self._services['prediction_service']
    
    def reset(self):
        """Reset all services (useful for testing)."""
        self._services.clear()


class ContainerFactory:
    """Factory for creating service containers."""
    
    _default_container: ServiceContainer = None
    
    @classmethod
    def get_default_container(cls) -> ServiceContainer:
        """
        Get default application container (Singleton).
        
        Returns:
            Default service container
        """
        if cls._default_container is None:
            cls._default_container = ServiceContainer()
        return cls._default_container
    
    @classmethod
    def create_container(cls, config: ApplicationConfig = None) -> ServiceContainer:
        """
        Create new container with custom configuration.
        
        Args:
            config: Custom configuration
            
        Returns:
            New service container
        """
        return ServiceContainer(config=config)
    
    @classmethod
    def reset_default(cls):
        """Reset default container (useful for testing)."""
        if cls._default_container is not None:
            cls._default_container.reset()
        cls._default_container = None


def get_prediction_service() -> IPredictionService:
    """
    Convenience function to get prediction service.
    
    This is the main entry point for the application.
    
    Returns:
        Prediction service with all dependencies resolved
    """
    container = ContainerFactory.get_default_container()
    return container.get_prediction_service()

