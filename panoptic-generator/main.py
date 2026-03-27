"""Entry point: launches GUI (no args) or CLI (with args)."""

import sys


def main():
    if len(sys.argv) > 1:
        from cli import cli

        cli()
    else:
        try:
            from gui.main_window import run_gui

            run_gui()
        except ImportError:
            from cli import cli

            cli(["--help"])


if __name__ == "__main__":
    main()
