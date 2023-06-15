"""
Installs Python packages from a requirements.txt file using pip.

This script uses the subprocess module to run pip and install the packages specified in the requirements.txt file.

Requires:
    Python version 3.9 or later
    pip package manager

Usage:
    Place the requirements.txt file in the same directory as this script.
    Run the script.

Note:
    This script must be run with administrative privileges if installing system-wide packages.
"""

try:
    import subprocess  # Subprocess management
    import signal  # Set handlers for asynchronous events
    import time as tim  # Time access and conversions
    from typing import Optional  # Support for type hints
except ImportError as L_err:
    print("Chyba v načtení standardní knihovny: {0}".format(L_err))
    exit(1)
except Exception as e:
    print(f"Jiná chyba v načtení standardní knihovny: {e}")
    exit(1)

try:
    with open('requirements.txt') as f:
        packages = f.read().splitlines()
except FileNotFoundError:
    print("Soubor 'requirements.txt' nebyl nalezen.")
    exit(1)
except Exception as e:
    print(f"Neočekávaná chyba: {e}")
    exit(1)


def handle_keyboard_interrupt(frame: Optional[object] = None, signal: Optional[object] = None) -> None:
    """
    A function for handling keyboard interrupt signal.

    :param frame: Not used.
    :param signal: Not used.
    :return: None
    """
    print("\nUkončeno stiskem klávesy Ctrl+C nebo tlačítkem Stop")
    exit()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_keyboard_interrupt)
    print(f"Instalace {len(packages)} balíčků z requirements.txt...")
    print("Větší balíčky trvají delší dobu, dle rychlosti připojení apod.")
    start_time = tim.time()
    for i, package in enumerate(packages):
        print(f"Instalace {package}...")
        try:
            subprocess.check_call(["pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"Chyba při instalaci balíčku {package}: {e}")
            exit(1)
        except Exception as e:
            print(f"Jiná chyba při instalaci balíčku {package}: {e}")
            exit(1)

        # Počítání procent dokončení instalace
        percentage_complete = ((i+1) / len(packages)) * 100
        time_elapsed = tim.time() - start_time
        estimated_time_remaining = time_elapsed * ((100 - percentage_complete) / percentage_complete)
        print(f"{percentage_complete:.2f}% hotovo ({int(estimated_time_remaining)}s zbývá)...")

    print("Všechny balíčky se stáhnuly úspěšně!")
