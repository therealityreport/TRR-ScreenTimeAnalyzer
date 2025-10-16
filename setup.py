"""Setup script for Screen Time Analyzer package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README if it exists
readme_file = Path(__file__).parent / "README_HARVEST_CHANGES.md"
long_description = ""
if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")

setup(
    name="screentime-analyzer",
    version="0.1.0",
    description="Automated screen time measurement for video content",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Screen Time Analyzer Team",
    packages=find_packages(exclude=["tests*", "docs*"]),
    python_requires=">=3.9,<3.12",
    install_requires=[
        "insightface>=0.7.3",
        "onnxruntime>=1.16.3",  # 1.16.3 for Mac CoreML compatibility
        "ultralytics>=8.3.0",
        "opencv-python>=4.9.0,<4.12",  # Lock to 4.11.x for NumPy 1.x compatibility
        "numpy>=1.26.0,<2.0",  # Lock to 1.x for onnxruntime compatibility
        "pandas>=2.2.0",
        "pyarrow>=15.0.0",
        "tqdm>=4.66.0",
        "pillow>=10.3.0",
        "pyyaml>=6.0.0",
        "scipy",
        "scikit-learn",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black",
            "flake8",
            "mypy",
        ],
        "ui": [
            "streamlit>=1.37.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "screentime-harvest=scripts.harvest_faces:main",
            "screentime-tracker=scripts.run_tracker:main",
            "screentime-diagnose=scripts.diagnose_harvest:main",
            "screentime-validate=scripts.validate_harvest:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
