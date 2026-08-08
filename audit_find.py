"""Temporary helper: list every line in the given docs mentioning the federated scale."""
import re
import sys

pattern = re.compile(
    r"4 clients|15 rounds|4-client|15-round|15 rows|num_clients.*4|num_rounds.*15|"
    r"min_available_clients|min_fit_clients|min_evaluate_clients|local_epochs|"
    r"Revision|revision|four clients|fifteen"
)

for path in sys.argv[1:]:
    lines = open(path, encoding="utf-8").read().splitlines()
    print(f"########## {path} ##########")
    for i, line in enumerate(lines, start=1):
        if pattern.search(line):
            print(f"{i}: {line}")
    print()
