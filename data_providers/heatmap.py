class HeatmapProvider:
    """
    Placeholder scaffold for an external heatmap/CVD provider.
    Implement `get_cvd` or `get_heatmap` to return data structures used by the strategy.
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key

    def get_heatmap(self, inst_id: str, timeframe: str):
        """Return heatmap-like structure or raise NotImplementedError when not available."""
        raise NotImplementedError("Heatmap provider not configured. Provide implementation or API key.")

    def get_cvd(self, inst_id: str, timeframe: str):
        """Return cumulative volume delta or None if not available."""
        raise NotImplementedError("CVD not implemented in placeholder provider.")
