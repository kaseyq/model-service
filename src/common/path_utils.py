import os

class PathUtil:
    """Utility class to provide path prefixes based on execution environment."""
    
    @staticmethod
    def is_docker() -> bool:
        """Check if running inside a Docker container."""
        # Check for Docker-specific environment variable
        if os.getenv('DOCKER_CONTAINER', False):
            return True
        # Check /proc/self/cgroup for Docker or containerd signatures
        try:
            with open('/proc/self/cgroup', 'r') as f:
                return any('docker' in line or 'containerd' in line for line in f)
        except FileNotFoundError:
            return False
    
    @staticmethod
    def get_base_path() -> str:
        """Return base path: '/app/' for Docker, './' for host."""
        return '/app/' if PathUtil.is_docker() else './'
    
    @staticmethod
    def get_path(relative_path: str) -> str:
        """Return full path with appropriate prefix."""
        base_path = PathUtil.get_base_path()
        # Remove leading './' or '/' to avoid duplication
        clean_path = relative_path.lstrip('./').lstrip('/')
        return os.path.join(base_path, clean_path)