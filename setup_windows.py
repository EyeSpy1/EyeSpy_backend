import subprocess
import sys
import os
import platform
import logging
from typing import List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WindowsSetup:
    def __init__(self):
        self.python_version = sys.version.split()[0]
        self.is_64bits = sys.maxsize > 2**32
        self.platform = platform.system()
        
    def check_requirements(self) -> List[str]:
        """Check if all required build tools are available"""
        missing_requirements = []
        
        # Check for Visual C++ Build Tools
        try:
            import distutils.msvc9compiler
        except ImportError:
            missing_requirements.append("Visual C++ Build Tools")
            
        # Check for CMake
        try:
            subprocess.run(["cmake", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            missing_requirements.append("CMake")
            
        return missing_requirements
        
    def install_base_requirements(self) -> bool:
        """Install base requirements needed for building"""
        try:
            # Install cmake first
            subprocess.run([
                sys.executable, 
                "-m", 
                "pip", 
                "install", 
                "--upgrade",
                "cmake==3.25.0"
            ], check=True)
            
            # Install setuptools and wheel
            subprocess.run([
                sys.executable,
                "-m",
                "pip",
                "install",
                "setuptools==65.0.0",
                "wheel==0.38.0"
            ], check=True)
            
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error installing base requirements: {str(e)}")
            return False
            
    def install_dlib(self) -> bool:
        """Install dlib with specific configuration"""
        try:
            # First ensure numpy is installed
            subprocess.run([
                sys.executable,
                "-m",
                "pip",
                "install",
                "numpy==1.23.4"
            ], check=True)
            
            # Install dlib
            subprocess.run([
                sys.executable,
                "-m",
                "pip",
                "install",
                "dlib==19.24.1"
            ], check=True)
            
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error installing dlib: {str(e)}")
            return False
            
    def install_requirements(self) -> bool:
        """Install all other requirements"""
        try:
            requirements = [
                "streamlit==1.20.0",
                "uuid==1.30",
                "opencv-python-headless",
                "imutils==0.5.4",
                "pyttsx3==2.90"
            ]
            
            for req in requirements:
                subprocess.run([
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    req
                ], check=True)
                
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error installing requirements: {str(e)}")
            return False
            
    def setup(self):
        """Run the complete setup process"""
        logger.info("Starting Windows setup...")
        
        # Check Python version
        logger.info(f"Python version: {self.python_version}")
        if not (3, 8) <= sys.version_info < (3, 12):
            logger.error("Python version must be between 3.8 and 3.11")
            return False
            
        # Check system requirements
        missing_reqs = self.check_requirements()
        if missing_reqs:
            logger.error(f"Missing requirements: {', '.join(missing_reqs)}")
            logger.error("Please install Visual Studio Build Tools and CMake")
            return False
            
        # Install base requirements
        logger.info("Installing base requirements...")
        if not self.install_base_requirements():
            return False
            
        # Install dlib
        logger.info("Installing dlib...")
        if not self.install_dlib():
            return False
            
        # Install other requirements
        logger.info("Installing other requirements...")
        if not self.install_requirements():
            return False
            
        logger.info("Setup completed successfully!")
        return True

def main():
    setup = WindowsSetup()
    if setup.setup():
        print("\nSetup completed successfully! You can now run your Streamlit app.")
        print("Run: streamlit run your_app.py")
    else:
        print("\nSetup failed. Please check the error messages above.")
        print("\nMake sure you have:")
        print("1. Visual Studio Build Tools installed")
        print("2. CMake installed")
        print("3. Python 3.8-3.11")

if __name__ == "__main__":
    main()