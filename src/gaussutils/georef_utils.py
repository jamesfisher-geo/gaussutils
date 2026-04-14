import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def save_georef_transform(
    ecef_to_enu: np.ndarray,
    output_path: Path,
) -> None:
    """Save a JSON sidecar describing the ENU coordinate system of the model.

    Args:
        enu_to_ecef: 4x4 ECEF→ENU transform matrix.
        output_path: Output .json path.
    """
    data = {
        "coordinate_system": "ENU",
        "enu_to_ecef_matrix": np.linalg.inv(ecef_to_enu).tolist(),
        "note": (
            "Model coordinates are ENU meters. "
            "Apply enu_to_ecef_matrix to convert the outputs to ECEF (EPSG:4978)."
        ),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2))
    logger.info(f"Saved georef transform sidecar to {output_path}")
