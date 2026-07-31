"""KEA model-ingest frontend: float model -> quantized integer graph IR.

Public surface:

``kea_frontend.ir``          the ``.kgraph.json`` / ``.npz`` format
``kea_frontend.apply_scale`` the requantization primitive (normative)
``kea_frontend.reference``   the bit-exact integer reference interpreter
``kea_frontend.quant``       observers, scale derivation, BN folding
``kea_frontend.builder``     dual-backend network builder
``kea_frontend.onnx_ingest`` ONNX -> KGraph

See ``docs/FRONTEND.md`` and ``docs/QUANTIZATION.md``.
"""

__version__ = "0.1.0"

from .ir import KGRAPH_VERSION, KGraph, KGraphError, Node, QuantParams, Tensor  # noqa: F401

__all__ = ["KGraph", "KGraphError", "Node", "QuantParams", "Tensor", "KGRAPH_VERSION"]
