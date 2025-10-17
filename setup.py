#!/usr/bin/env python3
"""
Setup script for Tokenomics AI Platform
Run this script to set up the project environment
"""

import os
import sys
import subprocess
import platform

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    print("🚀 Setting up Tokenomics AI Platform")
    print("=" * 50)

    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 8):
        print(f"❌ Python {python_version.major}.{python_version.minor} detected. Please use Python 3.8 or higher.")
        sys.exit(1)

    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro} detected")

    # Create virtual environment
    if not os.path.exists('venv'):
        if not run_command('python -m venv venv', 'Creating virtual environment'):
            sys.exit(1)
    else:
        print("ℹ️  Virtual environment already exists")

    # Determine activation command
    if platform.system() == 'Windows':
        activate_cmd = 'venv\\Scripts\\activate'
        pip_cmd = 'python -m pip install --upgrade pip'
    else:
        activate_cmd = 'source venv/bin/activate'
        pip_cmd = 'python -m pip install --upgrade pip'

    # Install requirements
    if os.path.exists('requirements.txt'):
        pip_install_cmd = f'{activate_cmd} && python -m pip install -r requirements.txt'
        if not run_command(pip_install_cmd, 'Installing Python dependencies'):
            sys.exit(1)
    else:
        print("⚠️  requirements.txt not found. Please ensure it exists.")

    # Check if data directory exists and has required files
    data_dir = 'data/elliptic'
    required_files = [
        'elliptic_txs_features.csv',
        'elliptic_txs_edgelist.csv',
        'elliptic_txs_classes.csv'
    ]

    if os.path.exists(data_dir):
        missing_files = []
        for file in required_files:
            if not os.path.exists(os.path.join(data_dir, file)):
                missing_files.append(file)

        if missing_files:
            print(f"⚠️  Missing data files in {data_dir}: {', '.join(missing_files)}")
            print("   GNN features will be limited. Download from: https://www.kaggle.com/ellipticco/elliptic-data-set")
        else:
            print("✅ Elliptic dataset files found")
    else:
        print(f"⚠️  Data directory {data_dir} not found. GNN features will be limited.")

    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nTo run the platform:")
    print("1. Activate virtual environment:")
    if platform.system() == 'Windows':
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    print("2. Start the API server:")
    print("   python api/simple_app.py")
    print("3. Open browser to: http://localhost:3000")
    print("\n📚 For detailed instructions, see README.md")

if __name__ == '__main__':
    main()