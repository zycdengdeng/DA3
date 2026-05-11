"""Allow ``python -m lidar_anchored_depth.cli ...`` as an alias for
the installed ``lad`` script. Useful when the package is on PYTHONPATH
but not pip-installed (e.g. during local development)."""

from lidar_anchored_depth.cli import main

raise SystemExit(main())
