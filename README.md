![Python](https://img.shields.io/badge/Python-3.x-blue)
![Status](https://img.shields.io/badge/Status-Prototype-orange)

# Merge Tactics --- Deck Optimizer (Synergy Pairs)

A Python prototype inspired by **Clash Royale-style deck building** and
**auto-battler synergy systems**.

The objective is to automatically build the strongest possible team
(deck) by maximizing the number of **activated traits** --- meaning
traits that appear **at least twice** in a team.

The optimizer supports optional gameplay constraints such as: - Locked
cards (must be included) - Banned cards (must be excluded) - Maximum
elixir budget - Pre-activated starting traits

------------------------------------------------------------------------

## Project Overview

In synergy-based games, team strength often depends on activating
multiple trait combinations.

This project models that logic and searches for the optimal team
composition using:

-   A trait activation system (threshold ≥ 2)
-   A beam search strategy to efficiently explore combinations
-   A scoring system prioritizing synergy breadth over raw power

------------------------------------------------------------------------

## Features

-   Trait-based synergy activation
-   Beam search optimization
-   Optional constraints:
    -   `--locked`
    -   `--banned`
    -   `--max-elixir`
    -   `--initial-traits`
-   Clean modular architecture
-   CSV-based dataset (separated from code)

------------------------------------------------------------------------

## Installation

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## How to Run

Basic usage:

``` bash
python -m src.cli
```

------------------------------------------------------------------------

## CLI Examples

### Force a specific card into the team

``` bash
python -m src.cli --locked "Knight"
```

### Ban certain cards

``` bash
python -m src.cli --banned "Wizard,Witch"
```

### Start with pre-activated traits

``` bash
python -m src.cli --initial-traits "Goblin,Clan"
```

### Apply an elixir constraint

``` bash
python -m src.cli --max-elixir 20
```

### Combine constraints

``` bash
python -m src.cli --locked "Knight" --banned "Wizard" --initial-traits "Goblin" --max-elixir 22
```

------------------------------------------------------------------------

## Example Output

    === Best Team ===
    pairs>=2: 6
    team: ['Knight', 'Archers', 'Goblins', 'Spear Goblins', 'Barbarians', 'Skeleton Dragons']
    active traits: ['Noble', 'Clan', 'Goblin', 'Ranger', 'Juggernaut', 'Brawler']

------------------------------------------------------------------------

## Scoring Logic

The scoring system follows a clear priority:

1.  Maximize number of activated traits (≥2 occurrences)
2.  Minimize total elixir cost (tie-break)
3.  Base power is included for informational purposes

This encourages **breadth-based synergy strategies**, similar to
auto-battler game mechanics.

------------------------------------------------------------------------

## Data Structure

Cards are defined in:

data/cards.csv

Required fields:

-   `card`
-   `elixir`
-   `base_power`
-   `traits` (semicolon-separated)

You can expand the card pool by simply editing the CSV file.

------------------------------------------------------------------------

## Project Structure

```
personal-game-merge-tactics/
├── data/
│   └── cards.csv
├── src/
│   ├── models.py
│   ├── io_data.py
│   ├── scoring.py
│   ├── search.py
│   └── cli.py
├── requirements.txt
└── README.md
```

------------------------------------------------------------------------

## Technical Highlights

-   Python 3
-   Pandas / NumPy
-   Custom beam search implementation
-   Modular architecture
-   CLI interface via argparse

------------------------------------------------------------------------

## Why This Project Matters

This project demonstrates:

-   Combinatorial optimization
-   Heuristic search design
-   Clean code modularization
-   Separation of data and logic
-   CLI tool development
-   Game mechanics modeling

It bridges algorithmic thinking with practical system design.

------------------------------------------------------------------------

## Project Status

Prototype --- functional core complete.

Planned improvements:

-   Export results to CSV / JSON
-   Advanced trait thresholds (2 / 4 / 6 scaling)
-   Performance benchmarking
-   Unit tests
-   Web interface or visualization layer

------------------------------------------------------------------------

## Author

Mathieu Alassoeur\
Data Analyst \| Business Intelligence
