import json
from pathlib import Path


class Config:
    """Configuration class for Bandsalat project paths."""

    _instance: "Config | None" = None

    def __init__(self, config_path: str | Path | None = None):
        """
        Initialize Config with optional path to config.json.

        Args:
            config_path: Optional path to config.json. If not provided,
                        will search in current directory and parent directories.
        """
        self._config_data: dict | None = None
        self._project_root: Path | None = None
        self._config_file_path: Path | None = None
        self._load_config(config_path)

    def _load_config(self, config_path: str | Path | None = None) -> None:
        """Load configuration from JSON file."""
        if config_path is not None:
            config_file = Path(config_path)
            if not config_file.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            self._config_file_path = config_file.resolve()
            self._project_root = self._config_file_path.parent
        else:
            config_file = self._find_config_file()
            if config_file is None:
                raise FileNotFoundError(
                    "Could not find config.json in current directory or parent directories"
                )
            self._config_file_path = config_file
            self._project_root = config_file.parent

        with open(self._config_file_path, "r") as f:
            self._config_data = json.load(f)

    @staticmethod
    def _find_config_file() -> Path | None:
        """Search for config.json in current directory and parent directories."""
        search_paths = [
            Path.cwd(),
            Path.cwd().parent,
            Path(__file__).parent.parent.parent,
        ]

        for search_path in search_paths:
            config_file = search_path / "config.json"
            if config_file.exists():
                return config_file.resolve()

        return None

    @classmethod
    def get_instance(cls, config_path: str | Path | None = None) -> "Config":
        """Get or create singleton instance of Config."""
        if cls._instance is None:
            cls._instance = cls(config_path)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        cls._instance = None

    def _get_path(self, key: str) -> Path:
        """Get a path from config, resolved relative to project root."""
        relative_path = self._config_data["paths"].get(key)
        if relative_path is None:
            raise KeyError(f"Path key not found in config: {key}")
        return self._project_root / relative_path

    def get_project_root(self) -> Path:
        """Get the project root directory."""
        return self._project_root

    def get_data_dir(self) -> Path:
        """Get the main data directory."""
        return self._get_path("data")

    def get_models_dir(self) -> Path:
        """Get the models directory."""
        return self._get_path("models")

    def get_audio_dir(self) -> Path:
        """Get the audio directory."""
        return self._get_path("audio")

    def get_audio_assets_dir(self) -> Path:
        """Get the audio assets directory (delimiters, spoken digits, etc.)."""
        return self._get_path("audio_assets")

    def get_datasets_dir(self) -> Path:
        """Get the datasets directory (contains individual dataset folders)."""
        return self._get_path("datasets")

    def get_datasets_workdir(self) -> Path:
        """Get the datasets working directory (for compilation, processing)."""
        return self._get_path("datasets_workdir")

    def get_playground_dir(self) -> Path:
        """Get the playground directory (for demo notebooks and experimentation)."""
        return self._get_path("playground")

    def print_paths(self) -> None:
        """Print all configured paths for debugging."""
        print("Bandsalat Configuration:")
        print(f"  Project root:      {self.get_project_root()}")
        print(f"  Data:              {self.get_data_dir()}")
        print(f"  Models:            {self.get_models_dir()}")
        print(f"  Audio:             {self.get_audio_dir()}")
        print(f"  Audio assets:      {self.get_audio_assets_dir()}")
        print(f"  Datasets:          {self.get_datasets_dir()}")
        print(f"  Datasets workdir:  {self.get_datasets_workdir()}")
        print(f"  Playground:        {self.get_playground_dir()}")
