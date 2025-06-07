import sys
import subprocess

def print_banner():
    """Pretty ASCII banner (installs *pyfiglet* the first time)."""
    try:
        import pyfiglet
    except ImportError:
        print("pyfiglet not found. Installing pyfiglet…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyfiglet"])
        import pyfiglet
    ascii_banner = pyfiglet.figlet_format("LLMRipper")
    print(ascii_banner)
    print("Created by Alican Kiraz – v2.0")