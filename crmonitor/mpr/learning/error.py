class FeatureExtractionError(Exception):
    def __init__(self, name: str, hint: str | None = None) -> None:
        msg = f"Failed to extract feature variable {name}"
        if hint is not None:
            msg += f": {hint}"
        super().__init__(msg)
