#!/usr/bin/env python3
"""
Evaluation utilities for self-supervised and supervised learning.
"""

from .self_supervised_metrics import (
    evaluate_link_prediction,
    create_negative_samples_pyg,
)

__all__ = [
    'evaluate_link_prediction',
    'create_negative_samples_pyg',
]
