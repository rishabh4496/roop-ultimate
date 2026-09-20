class ProcessEntry:
    def __init__(self, filename: str, start: int, end: int, fps: float,
                 media_id: str | None = None):
        self.filename = filename
        # This is the identity of the target media item, not its path/name or
        # current position in the target queue.  The API fills it when loading
        # legacy entries that were created without the optional argument.
        self.media_id = media_id
        self.finalname = None
        self.startframe = start
        self.endframe = end
        self.fps = fps
