class InteractionEngine:
    """Single safety and routing gate for all active input providers."""

    def __init__(self, actions, minimum_confidence=0.72, on_pause=None):
        self.actions = actions
        self.minimum_confidence = minimum_confidence
        self.on_pause = on_pause
        self.paused = True
        self.enabled_sources = set()

    def start(self, sources=()):
        self.enabled_sources = set(sources)
        self.paused = False

    def pause(self):
        self.paused = True
        if self.on_pause:
            self.on_pause()

    def dispatch(self, event):
        if event.action == "pause":
            self.pause()
            return True
        if self.paused or event.source not in self.enabled_sources:
            return False
        if not 0.0 <= event.confidence <= 1.0 or event.confidence < self.minimum_confidence:
            return False
        handlers = {
            "move_pointer": self.actions.move_pointer,
            "click": self.actions.click,
            "scroll": self.actions.scroll,
            "type_text": self.actions.type_text,
            "press_key": self.actions.press_key,
            "open_application": self.actions.open_application,
            "activate_window": self.actions.activate_window,
            "click_control": self.actions.click_control,
        }
        handler = handlers.get(event.action)
        if not handler:
            return False
        try:
            handler(**event.payload)
            return True
        except Exception:
            return False
