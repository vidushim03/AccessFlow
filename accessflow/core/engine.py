import logging

logger = logging.getLogger(__name__)


class InteractionEngine:
    """Single safety and routing gate for all active input providers."""

    # Actions that can block for seconds (typing, UI automation) are handed to
    # the action thread so they never freeze the interface or delay pause.
    DEFERRED_ACTIONS = frozenset({
        "type_text", "press_key", "open_application", "activate_window",
        "click_control", "click_target",
    })

    # Free dictation is text the user can see and correct, so it tolerates a
    # lower score than a command that moves or clicks something. Matches the
    # voice worker's dictation threshold.
    ACTION_CONFIDENCE = {
        "type_text": 0.60,
        "click": 0.65,
        "click_target": 0.65,
    }

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
        handler_names = {
            "move_pointer": "move_pointer",
            "click": "click",
            "click_target": "click_target",
            "scroll": "scroll",
            "type_text": "type_text",
            "press_key": "press_key",
            "open_application": "open_application",
            "activate_window": "activate_window",
            "click_control": "click_control",
        }
        handler_name = handler_names.get(event.action)
        handler = getattr(self.actions, handler_name, None) if handler_name else None
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
