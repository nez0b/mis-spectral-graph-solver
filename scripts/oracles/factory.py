"""
Oracle factory for creating oracle adapters.
"""

from typing import Dict, Type, Any, Optional, List
from .base import OracleAdapter, OracleConfig
from .jax_pgd_adapter import JAXPGDAdapter, JAXPGDConfig
from .dirac_adapter import DiracAdapter, DiracConfig


class OracleFactory:
    """
    Factory class for creating oracle adapters.
    
    Provides a unified interface for creating different types of oracle adapters
    with appropriate configurations and availability checking.
    """
    
    # Registry of available oracle types
    _oracle_registry: Dict[str, Dict[str, Any]] = {
        'jax-pgd': {
            'adapter_class': JAXPGDAdapter,
            'config_class': JAXPGDConfig,
            'description': 'JAX Projected Gradient Descent with multiple restarts'
        },
        'dirac': {
            'adapter_class': DiracAdapter,
            'config_class': DiracConfig,
            'description': 'Dirac-3 quantum annealing solver'
        }
    }
    
    @classmethod
    def get_available_oracles(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get information about available oracle types.
        
        Returns:
            Dictionary mapping oracle names to their information and availability
        """
        result = {}
        for oracle_name, oracle_info in cls._oracle_registry.items():
            adapter_class = oracle_info['adapter_class']
            
            # Check if dependencies are available by creating a minimal config
            # and testing adapter availability
            try:
                if oracle_name == 'jax-pgd':
                    config = JAXPGDConfig(num_restarts=1)  # Minimal config for testing
                elif oracle_name == 'dirac':
                    config = DiracConfig(num_samples=1)  # Minimal config for testing
                else:
                    config = None
                
                # Test availability without full initialization
                available = hasattr(adapter_class, 'is_available') and \
                           getattr(adapter_class, '_test_availability', lambda: True)()
                
                # More robust availability check
                try:
                    temp_adapter = adapter_class.__new__(adapter_class)  # Create without __init__
                    available = temp_adapter.is_available
                except:
                    available = False
                    
            except Exception:
                available = False
            
            result[oracle_name] = {
                'description': oracle_info['description'],
                'available': available,
                'config_class': oracle_info['config_class'].__name__
            }
        
        return result
    
    @classmethod
    def create_oracle(
        cls, 
        oracle_type: str, 
        config: Optional[OracleConfig] = None,
        verbose: bool = False,
        enable_refinement: bool = True,
        **kwargs
    ) -> OracleAdapter:
        """
        Create an oracle adapter of the specified type.
        
        Args:
            oracle_type: Type of oracle ('jax-pgd', 'dirac')
            config: Pre-configured oracle configuration object
            verbose: Whether to enable verbose output
            enable_refinement: Whether to enable superposition refinement (default: True)
            **kwargs: Configuration parameters (used if config is None)
            
        Returns:
            Configured oracle adapter instance
            
        Raises:
            ValueError: If oracle_type is not supported
            ImportError: If oracle dependencies are not available
            RuntimeError: If oracle cannot be initialized
        """
        if oracle_type not in cls._oracle_registry:
            available_types = list(cls._oracle_registry.keys())
            raise ValueError(
                f"Unknown oracle type: {oracle_type}. "
                f"Available types: {available_types}"
            )
        
        oracle_info = cls._oracle_registry[oracle_type]
        adapter_class = oracle_info['adapter_class']
        config_class = oracle_info['config_class']
        
        # Create configuration if not provided
        if config is None:
            try:
                config = config_class(**kwargs)
            except Exception as e:
                raise ValueError(f"Failed to create {oracle_type} configuration: {e}")
        
        # Validate configuration type
        if not isinstance(config, config_class):
            raise TypeError(
                f"Configuration must be of type {config_class.__name__} "
                f"for oracle type {oracle_type}, got {type(config).__name__}"
            )
        
        # Create adapter
        try:
            adapter = adapter_class(config, verbose=verbose, enable_refinement=enable_refinement)
            return adapter
        except ImportError as e:
            available_oracles = cls.get_available_oracles()
            if not available_oracles[oracle_type]['available']:
                raise ImportError(
                    f"Oracle {oracle_type} is not available due to missing dependencies. "
                    f"Error: {e}"
                )
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to initialize {oracle_type} oracle: {e}")
    
    @classmethod
    def create_default_config(cls, oracle_type: str, **overrides) -> OracleConfig:
        """
        Create a default configuration for the specified oracle type.
        
        Args:
            oracle_type: Type of oracle ('jax-pgd', 'dirac')
            **overrides: Configuration parameters to override defaults
            
        Returns:
            Default configuration with any specified overrides
            
        Raises:
            ValueError: If oracle_type is not supported
        """
        if oracle_type not in cls._oracle_registry:
            available_types = list(cls._oracle_registry.keys())
            raise ValueError(
                f"Unknown oracle type: {oracle_type}. "
                f"Available types: {available_types}"
            )
        
        config_class = cls._oracle_registry[oracle_type]['config_class']
        return config_class(**overrides)
    
    @classmethod
    def get_oracle_info(cls, oracle_type: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific oracle type.
        
        Args:
            oracle_type: Type of oracle to get information about
            
        Returns:
            Dictionary with oracle information
            
        Raises:
            ValueError: If oracle_type is not supported
        """
        if oracle_type not in cls._oracle_registry:
            available_types = list(cls._oracle_registry.keys())
            raise ValueError(
                f"Unknown oracle type: {oracle_type}. "
                f"Available types: {available_types}"
            )
        
        oracle_info = cls._oracle_registry[oracle_type]
        config_class = oracle_info['config_class']
        
        # Get configuration parameters by inspecting the config class __init__
        import inspect
        config_signature = inspect.signature(config_class.__init__)
        config_params = {
            name: {
                'default': param.default if param.default != inspect.Parameter.empty else None,
                'annotation': param.annotation if param.annotation != inspect.Parameter.empty else None
            }
            for name, param in config_signature.parameters.items()
            if name != 'self'
        }
        
        return {
            'type': oracle_type,
            'description': oracle_info['description'],
            'adapter_class': oracle_info['adapter_class'].__name__,
            'config_class': config_class.__name__,
            'config_parameters': config_params,
            'available': cls.get_available_oracles()[oracle_type]['available']
        }
    
    @classmethod
    def list_available_oracles(cls, verbose: bool = False) -> List[str]:
        """
        List all available oracle types.
        
        Args:
            verbose: If True, includes unavailable oracles with reasons
            
        Returns:
            List of available oracle type names
        """
        available_oracles = cls.get_available_oracles()
        
        if verbose:
            result = []
            for oracle_type, info in available_oracles.items():
                status = "AVAILABLE" if info['available'] else "UNAVAILABLE"
                result.append(f"{status} {oracle_type}: {info['description']}")
            return result
        else:
            return [
                oracle_type for oracle_type, info in available_oracles.items()
                if info['available']
            ]


# Add test availability methods for clean availability checking
def _test_jax_pgd_availability() -> bool:
    """Test if JAX-PGD dependencies are available."""
    try:
        from .jax_pgd_adapter import JAX_AVAILABLE
        return JAX_AVAILABLE
    except:
        return False

def _test_dirac_availability() -> bool:
    """Test if Dirac dependencies are available."""
    try:
        from .dirac_adapter import QCI_AVAILABLE, GRAPH_TO_OMEGA_AVAILABLE
        return QCI_AVAILABLE and GRAPH_TO_OMEGA_AVAILABLE
    except:
        return False

# Add test methods to adapter classes
JAXPGDAdapter._test_availability = staticmethod(_test_jax_pgd_availability)
DiracAdapter._test_availability = staticmethod(_test_dirac_availability)