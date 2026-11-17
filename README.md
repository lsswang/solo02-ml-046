# Occupancy evidence and route decisions

Run `python demo.py` with Python 3.11; the starter uses only the standard library.
It contains a working independent-cell inverse sensor update and ordinary
Dijkstra using known blocked cells. It has no joint first-hit posterior or
chance-constrained planner. Synthetic fixtures describe cells by integer IDs;
edge cell lists already describe which uncertain cells a traversal visits.
The task text supplies the requested model and numerical contracts.
