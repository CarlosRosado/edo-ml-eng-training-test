import os
import pytest
from src.utils.utils import create_directory

@pytest.fixture
def test_dir(tmpdir):
    """Fixture to create a temporary test directory."""
    return str(tmpdir.mkdir("test_dir"))

def test_create_directory_creates_new_directory(test_dir):
    """Test that create_directory creates a new directory."""
    new_dir = os.path.join(test_dir, "new_directory")
    assert not os.path.exists(new_dir), "Directory should not exist before the test."
    create_directory(new_dir)
    assert os.path.exists(new_dir), "Directory should be created by create_directory."

def test_create_directory_does_not_fail_if_exists(test_dir):
    """Test that create_directory does not fail if the directory already exists."""
    create_directory(test_dir)  
    assert os.path.exists(test_dir), "Directory should still exist after calling create_directory."