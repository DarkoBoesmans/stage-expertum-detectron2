#!/usr/bin/env python3
import os
import re
import time
import sys
from datetime import datetime


def monitor_progress(log_file, total_iterations=10000):
    """Monitor training progress in the Detectron2 log file."""
    if not os.path.exists(log_file):
        print(f"Log file '{log_file}' niet gevonden.")
        return

    last_iter = 0

    while True:
        try:
            with open(log_file, "r") as f:
                content = f.read()

            # Zoek naar de laatste iteratie in de log
            iter_matches = re.findall(r"iter: (\d+)", content)
            if iter_matches:
                last_iter = int(iter_matches[-1])

            # Bereken percentage
            percentage = (last_iter / total_iterations) * 100

            # Clear terminal en toon voortgang
            os.system("clear")
            print(f"Detectron2 Training Voortgang:")
            print(f"================================")
            print(f"Huidige iteratie: {last_iter}/{total_iterations}")
            print(f"Percentage voltooid: {percentage:.2f}%")
            print(f"Geschatte tijd resterend: zie log voor details")

            if last_iter >= total_iterations:
                print("Training voltooid!")
                break

            # Controleer of de laatste update van het log-bestand recent is
            last_modified = os.path.getmtime(log_file)
            time_since_update = time.time() - last_modified

            if time_since_update > 600:  # 10 minuten
                print(
                    "\nWaarschuwing: Log file niet bijgewerkt in de afgelopen 10 minuten."
                )
                print("Het trainingsproces is mogelijk gestopt.")

            last_update = datetime.fromtimestamp(last_modified).strftime("%H:%M:%S")
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"\nLaatste update log-bestand: {last_update}")
            print(f"Huidige tijd: {current_time}")
            print(f"Druk op Ctrl+C om de monitor te stoppen.")

            # Wacht 5 seconden voor de volgende update
            time.sleep(5)

        except KeyboardInterrupt:
            print("\nMonitoring gestopt door gebruiker.")
            break
        except Exception as e:
            print(f"Fout bij monitoren: {e}")
            time.sleep(10)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        log_file = "output_small_batch.log"

    if len(sys.argv) > 2:
        total_iterations = int(sys.argv[2])
    else:
        total_iterations = 10000

    monitor_progress(log_file, total_iterations)
