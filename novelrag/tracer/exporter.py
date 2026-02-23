"""YAML exporter for trace data.

Converts the in-memory span tree to a human-readable YAML file that mirrors
the hierarchical structure of the trace.
"""

import logging
import os
from datetime import datetime
from pathlib import Path

import yaml

from novelrag.tracer.span import Span

logger = logging.getLogger(__name__)


class YAMLExporter:
    """Writes a completed trace tree to a YAML file.

    Parameters
    ----------
    output_dir:
        Directory where trace files will be written.  Created automatically
        if it does not exist.
    """

    def __init__(self, output_dir: Path | str) -> None:
        self.output_dir = Path(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    def export(self, root_span: Span, filename: str | None = None) -> Path:
        """Serialize *root_span* (the session span) to a YAML file.

        Returns the path to the written file.
        """
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trace_{ts}.yaml"

        path = self.output_dir / filename
        data = root_span.to_dict()

        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump(
                data,
                fh,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

        logger.info("Trace exported to %s", path)
        return path
