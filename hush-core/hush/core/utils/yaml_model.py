from pathlib import Path
from typing import Dict, Optional, Any

import yaml
from pydantic import BaseModel


class YamlModel(BaseModel):
    """Base class for yaml model"""

    extra_fields: Optional[Dict[str, Any]] = None

    @classmethod
    def read_yaml(
        cls, 
        file_path: Path, 
        encoding: str = "utf-8"
    ) -> Dict:
        """Read yaml file and return a dict
        
        Raises:
            FileNotFoundError: If the yaml file does not exist
            yaml.YAMLError: If the yaml file is malformed
            IOError: If there's an error reading the file
        """
        if not file_path.exists():
            raise FileNotFoundError(f"YAML file not found: {file_path}")
        
        try:
            with open(file_path, "r", encoding=encoding) as file:
                content = yaml.safe_load(file)
                return content if content is not None else {}
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {file_path}: {e}")
        except IOError as e:
            raise IOError(f"Error reading file {file_path}: {e}")

    @classmethod
    def from_yaml_file(
        cls, 
        file_path: Path
    ) -> "YamlModel":
        """Read yaml file and return a YamlModel instance
        
        Raises:
            FileNotFoundError: If the yaml file does not exist
            yaml.YAMLError: If the yaml file is malformed
            IOError: If there's an error reading the file
            ValidationError: If the yaml content doesn't match the model schema
        """        
        yaml_data = cls.read_yaml(file_path)
        try:
            return cls(**yaml_data)
        except Exception as e:
            raise ValueError(f"Error creating {cls.__name__} from {file_path}: {e}")

    def to_yaml_file(
        self, 
        file_path: Path, 
        encoding: str = "utf-8"
    ) -> None:
        """Dump YamlModel instance to yaml file
        
        Raises:
            IOError: If there's an error writing to the file
            yaml.YAMLError: If there's an error serializing to YAML
        """
        try:
            # Create parent directories if they don't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, "w", encoding=encoding) as file:
                yaml.dump(self.model_dump(), file, default_flow_style=False)
        except IOError as e:
            raise IOError(f"Error writing to file {file_path}: {e}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error serializing to YAML: {e}")

    def to_yaml_string(self) -> str:
        """Export YamlModel instance to a YAML string
        
        Returns:
            str: The YAML representation of the model
            
        Raises:
            yaml.YAMLError: If there's an error serializing to YAML
        """
        try:
            return yaml.dump(self.model_dump(), default_flow_style=False)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error serializing to YAML string: {e}")