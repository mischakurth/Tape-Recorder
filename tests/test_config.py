import json
import pytest
from pathlib import Path

from src.utils.config import Config


@pytest.fixture
def temp_config_dir(tmp_path):
    """Create a temporary directory with a valid config.json."""
    config_data = {
        "paths": {
            "data": "data",
            "models": "data/models",
            "audio": "data/audio",
            "audio_assets": "data/audio/assets",
            "datasets": "data/audio/datasets",
            "datasets_workdir": "data/audio/datasets-workdir"
        }
    }
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config_data))
    return tmp_path


@pytest.fixture
def minimal_config_dir(tmp_path):
    """Create a temporary directory with a minimal config.json."""
    config_data = {
        "paths": {
            "data": "my_data"
        }
    }
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config_data))
    return tmp_path


@pytest.fixture
def custom_paths_config_dir(tmp_path):
    """Create a temporary directory with custom paths."""
    config_data = {
        "paths": {
            "data": "custom/data/path",
            "models": "custom/models",
            "audio": "audio_files",
            "audio_assets": "audio_files/assets",
            "datasets": "audio_files/datasets",
            "datasets_workdir": "audio_files/workdir"
        }
    }
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config_data))
    return tmp_path


@pytest.fixture(autouse=True)
def reset_config_singleton():
    """Reset Config singleton before each test."""
    Config.reset()
    yield
    Config.reset()


class TestConfigInit:
    """Tests for Config initialization."""

    def test_init_with_explicit_path(self, temp_config_dir):
        """Test Config initialization with explicit config path."""
        config_path = temp_config_dir / "config.json"
        cfg = Config(config_path)

        assert cfg.get_project_root() == temp_config_dir

    def test_init_with_string_path(self, temp_config_dir):
        """Test Config initialization with string path."""
        config_path = str(temp_config_dir / "config.json")
        cfg = Config(config_path)

        assert cfg.get_project_root() == temp_config_dir

    def test_init_file_not_found(self, tmp_path):
        """Test Config raises error when config file not found."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            Config(tmp_path / "nonexistent.json")

    def test_init_auto_locate_from_project_root(self):
        """Test Config auto-locates config.json from project root."""
        cfg = Config()

        assert cfg.get_project_root().name == "xmas-hackathon-2025-bandsalat"
        assert (cfg.get_project_root() / "config.json").exists()


class TestConfigPaths:
    """Tests for Config path methods."""

    def test_get_data_dir(self, temp_config_dir):
        """Test get_data_dir returns correct path."""
        cfg = Config(temp_config_dir / "config.json")

        assert cfg.get_data_dir() == temp_config_dir / "data"

    def test_get_models_dir(self, temp_config_dir):
        """Test get_models_dir returns correct path."""
        cfg = Config(temp_config_dir / "config.json")

        assert cfg.get_models_dir() == temp_config_dir / "data" / "models"

    def test_get_audio_dir(self, temp_config_dir):
        """Test get_audio_dir returns correct path."""
        cfg = Config(temp_config_dir / "config.json")

        assert cfg.get_audio_dir() == temp_config_dir / "data" / "audio"

    def test_get_audio_assets_dir(self, temp_config_dir):
        """Test get_audio_assets_dir returns correct path."""
        cfg = Config(temp_config_dir / "config.json")

        assert cfg.get_audio_assets_dir() == temp_config_dir / "data" / "audio" / "assets"

    def test_get_datasets_dir(self, temp_config_dir):
        """Test get_datasets_dir returns correct path."""
        cfg = Config(temp_config_dir / "config.json")

        assert cfg.get_datasets_dir() == temp_config_dir / "data" / "audio" / "datasets"

    def test_get_datasets_workdir(self, temp_config_dir):
        """Test get_datasets_workdir returns correct path."""
        cfg = Config(temp_config_dir / "config.json")

        assert cfg.get_datasets_workdir() == temp_config_dir / "data" / "audio" / "datasets-workdir"

    def test_paths_are_absolute(self, temp_config_dir):
        """Test that all paths returned are absolute."""
        cfg = Config(temp_config_dir / "config.json")

        assert cfg.get_data_dir().is_absolute()
        assert cfg.get_models_dir().is_absolute()
        assert cfg.get_audio_dir().is_absolute()

    def test_missing_path_key_raises_error(self, minimal_config_dir):
        """Test that accessing missing path key raises KeyError."""
        cfg = Config(minimal_config_dir / "config.json")

        # data exists
        assert cfg.get_data_dir() == minimal_config_dir / "my_data"

        # models does not exist
        with pytest.raises(KeyError, match="Path key not found in config: models"):
            cfg.get_models_dir()


class TestCustomPaths:
    """Tests for Config with custom paths."""

    def test_custom_data_path(self, custom_paths_config_dir):
        """Test custom data path."""
        cfg = Config(custom_paths_config_dir / "config.json")

        assert cfg.get_data_dir() == custom_paths_config_dir / "custom" / "data" / "path"

    def test_custom_models_path(self, custom_paths_config_dir):
        """Test custom models path."""
        cfg = Config(custom_paths_config_dir / "config.json")

        assert cfg.get_models_dir() == custom_paths_config_dir / "custom" / "models"

    def test_custom_audio_paths(self, custom_paths_config_dir):
        """Test custom audio paths."""
        cfg = Config(custom_paths_config_dir / "config.json")

        assert cfg.get_audio_dir() == custom_paths_config_dir / "audio_files"
        assert cfg.get_audio_assets_dir() == custom_paths_config_dir / "audio_files" / "assets"
        assert cfg.get_datasets_dir() == custom_paths_config_dir / "audio_files" / "datasets"
        assert cfg.get_datasets_workdir() == custom_paths_config_dir / "audio_files" / "workdir"


class TestConfigSingleton:
    """Tests for Config singleton pattern."""

    def test_get_instance_creates_singleton(self, temp_config_dir):
        """Test get_instance creates and returns singleton."""
        config_path = temp_config_dir / "config.json"

        cfg1 = Config.get_instance(config_path)
        cfg2 = Config.get_instance()

        assert cfg1 is cfg2

    def test_reset_clears_singleton(self, temp_config_dir):
        """Test reset clears the singleton instance."""
        config_path = temp_config_dir / "config.json"

        cfg1 = Config.get_instance(config_path)
        Config.reset()
        cfg2 = Config.get_instance(config_path)

        assert cfg1 is not cfg2


class TestConfigWithRealProject:
    """Tests using the actual project config.json."""

    def test_load_project_config(self):
        """Test loading the actual project config."""
        cfg = Config()

        # Verify project root is correct
        assert cfg.get_project_root().name == "xmas-hackathon-2025-bandsalat"

        # Verify paths are constructed correctly
        assert cfg.get_data_dir().name == "data"
        assert cfg.get_models_dir().name == "models"
        assert cfg.get_audio_dir().name == "audio"

    def test_print_paths_does_not_raise(self):
        """Test print_paths executes without error."""
        cfg = Config()
        cfg.print_paths()  # Should not raise
