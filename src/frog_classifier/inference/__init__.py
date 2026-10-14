"""Batch detection: run a saved model over whole folders of recordings.

Nothing is re-exported here: the detection module imports PyTorch, while the
modules that rendering workers import must stay free of it. Import modules
directly.
"""
