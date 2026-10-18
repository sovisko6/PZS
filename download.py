"""Stažení databází PhysioNet do lokální složky (odolné proti výpadkům sítě).

Příklad:  python download.py drivedb voiced nsrdb
Poté spouštějte skripty s  --data-dir data  (rychlejší a spolehlivější než streamování).
Stahování lze kdykoli přerušit a spustit znovu, už stažené soubory se přeskočí.
Pozor: nsrdb a charisdb jsou velké (jednotky GB); u nich lze stáhnout jen část (--limit).
"""
import argparse
import time
from pathlib import Path

import wfdb

EXTRA = {"nsrdb": [".atr"]}


def fetch(db: str, target: Path, files: list[str], attempts: int = 8):
    for attempt in range(1, attempts + 1):
        try:
            wfdb.dl_files(db, str(target), files)
            return
        except Exception as exc:  # síťové chyby mají různé typy
            if attempt == attempts:
                raise
            print(f"   chyba ({type(exc).__name__}), pokus {attempt}/{attempts - 1} znovu ...")
            time.sleep(3 * attempt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("databases", nargs="+", choices=["drivedb", "charisdb", "nsrdb", "voiced"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--limit", type=int, default=None, help="Stáhnout jen prvních N záznamů")
    args = parser.parse_args()
    for db in args.databases:
        target = Path(args.data_dir) / db
        target.mkdir(parents=True, exist_ok=True)
        records = wfdb.get_record_list(db)[: args.limit]
        print(f"{db}: {len(records)} záznamů -> {target}")
        for i, rec in enumerate(records, 1):
            files = [rec + ext for ext in [".hea", ".dat"] + EXTRA.get(db, [])]
            files = [f for f in files if not (target / f).exists()]
            if files:
                print(f"  [{i}/{len(records)}] {rec}")
                fetch(db, target, files)
    print("Hotovo.")