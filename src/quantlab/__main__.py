"""`python -m quantlab` — alias de `python -m quantlab.cli`."""
import sys

from quantlab.cli import main

if __name__ == "__main__":
    sys.exit(main())
