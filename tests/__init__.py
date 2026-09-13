import warnings

# audioread's Python 3.13 shims for aifc and sunau emit DeprecationWarnings
# the first time librosa.load probes its backends. Import them once here,
# silently, so third-party import noise never appears in test output.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    try:
        # audioread's rawread backend is optional; its absence must not break tests.
        import audioread.rawread  # noqa: F401
    except ImportError:
        pass
