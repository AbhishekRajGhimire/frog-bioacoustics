"""The site staff app's services: deployments, scans, uploads, listening, and export.

Nothing is re-exported here; import modules directly. Only scan_job, which runs in
its own process, imports PyTorch, so the app itself starts quickly and stays small.
"""
