# Architecture

```text
Literature DB (Excel)
        |
        v
Schema validation (db_loader.py)
        |
        +-----------------------+
        |                       |
        v                       v
Exact literature match     Nearby-condition interpolation
(literature.py)            (literature.py / predictor.py)
        |                       |
        +-----------+-----------+
                    v
          Process-aware calculations
              (predictor.py)
                    |
          +---------+---------+
          |                   |
          v                   v
Profile visualization     Status diagnostics
(plotter.py)             (app.py)
          |
          v
Leave-One-Out model check
(validation.py)
```

The public showcase intentionally keeps the first screen simple. Detailed literature and validation information are moved into tabs so a reviewer can understand the research idea before reading model details.
