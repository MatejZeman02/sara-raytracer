import numpy as np
import src.settings as settings
settings.DEVICE = "cpu"
from src.main import _resolve_main
import sys
sys.argv.append("scenes/box-advanced/setup.json")
def main():
    _resolve_main()()
main()
