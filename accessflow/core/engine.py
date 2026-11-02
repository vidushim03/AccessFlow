import logging

logger = logging.getLogger(__name__)


class InteractionEngine:
    """Single safety and routing gate for all active input providers."""

    # Actions that can block for seconds (typing, UI automation) are handed to
    # the action thread so they never freeze the interface or delay pause.
    DEFERRED_ACTIONS = frozenset({
        "type_text", "press_key", "open_application", "activate_window", "click_control",
    })

    # Free dictation is text the user can see and correct, so it tolerates a
    # lower score than a command that moves or clicks something. Matches the
    # voice worker's dictation threshold.
    ACTION_CONFIDENCE = {"type_text": 0.60}

    def __init__(self, actions, minimum_confidence=0.72, on_pause=None,
                 on_error=None, executor=None):
        self.actions = actions
        self.minimum_confidence = minimum_confidence
        self.on_pause = on_pause
        self.on_error = on_error
        self.executor = executor
        self.paused = True
        self.enabled_sources = set()

    def start(self, sources=()):
        self.enabled_sources = set(sources)
        self.paused = False

    def pause(self):
        self.paused = True
        # Queued typing and UI automation must not run after the user asked to
        # stop; the action thread drops its backlog and stops mid-phrase.
        cancel = getattr(self.executor, "cancel", None)
        if cancel is not None:
            try:
                cancel()
            except Exception:
                logger.exception("Could not cancel queued actions")
        if self.on_pause:
            self.on_pause()

    def _report_failure(self, event, exc):
        logger.warning(
            "Action %s from %s failed: %s", event.action, event.source, exc)
        if self.on_error:
            try:
                self.on_error(f"{event.action.replace('_', ' ')} failed: {exc}")
            except Exception:
                pass

    def dispatch(self, event):
        if event.action == "pause":
            self.pause()
            return True
        if self.paused or event.source not in self.enabled_sources:
            return False
        threshold = self.ACTION_CONFIDENCE.get(event.action, self.minimum_confidence)
        if not 0.0 <= event.confidence <= 1.0 or event.confidence < threshold:
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
        if self.executor is not None and event.action in self.DEFERRED_ACTIONS:
            self.executor.submit(handler, event.payload)
            return True
        try:
            handler(**event.payload)
            return True
        except Exception as exc:
            self._report_failure(event, exc)
            return False
