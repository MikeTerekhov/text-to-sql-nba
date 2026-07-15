# Train Data Overview

This folder contains TSV files with subsets of SQL queries, each filtered based on specific criteria.

## Files and Descriptions

### `less_than_90.tsv`

- Contains queries where the total character length is **less than 90**.

### `queries_from_game.tsv`

- Includes queries where the **first table after the `FROM` clause is `game`**.

### `queries_from_other_stats.tsv`

- Contains queries where the **first table after `FROM` is `other_stats`**.

### `queries_from_team.tsv`

- Includes queries that **reference `team` immediately after `FROM`**.

### `queries_with_join.tsv`

- Contains queries that include the keyword **`JOIN`** (case-insensitive).

### `queries_without_join.tsv`

- Contains queries that **do not** include the keyword `JOIN`.
- Often simpler queries accessing a single table.

---
