#!/bin/bash

# Check if the argument is provided
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 -room <room_number>"
    exit 1
fi

# Run the Python scripts with the same arguments
python attacker.py "$@" > attacker.log 2>&1 &
python defender.py "$@" > defender.log 2>&1 &

wait
