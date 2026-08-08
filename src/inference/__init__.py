"""User-facing inference package for the IIoT Federated IDS project.

This package contains the read-only prediction path used by the dashboard's
CSV batch-inference tab. It trains nothing, fits nothing, and writes nothing to
``outputs/`` or ``data/`` — every encoder, scaler, label encoder and model it
uses is loaded from the artifacts already persisted by the experiment scripts.
"""
