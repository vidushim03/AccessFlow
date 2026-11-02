with open('accessflow/inputs/vision.py','r',newline='') as f:
    src = f.read()

old_block = '''                                    if pointer_active:
                                        sx, sy = self._smooth_pointer(
                                            "gaze", raw[0], raw[1], 6.0, absolute=True)
                                        self.event_ready.emit(InteractionEvent("gaze", "gaze", "move_pointer", 0.82, {
                                            "x": int(sx * (self.width - 1)), "y": int(sy * (self.height - 1))}))
                                if self.head_enabled and not gaze_active:
                                    if self.recenter_requested:
                                        self.head_tracker.reset()
                                        self._forget_source("head")
                                        self.recenter_requested = False
                                    head_feat = self._extract_head_feature(landmarks)
                                    if head_feat is not None:
                                        # Anchor to the current real pointer position on
                                        # first activation or after a recenter/takeover so
                                        # the pointer does not jump.
                                        anchor = None
                                        if self.head_tracker.smoothed is None:
                                            anchor = self._screen_pointer() or self.pointer_last or (0.5, 0.5)
                                        point = self._head_pointer(head_feat, anchor=anchor)
                                        if point is not None and pointer_active:
                                            self.pointer_source = "head"
                                            self.pointer_last = point
                                            self.event_ready.emit(InteractionEvent("head", "head", "move_pointer", 0.75, {
                                                "x": int(point[0] * (self.width - 1)), "y": int(point[1] * (self.height - 1))}))
                            elif now - last_face_seen > 3.0 and now - last_face_warning > 10.0:'''

new_block = '''                                    if pointer_active:
                                        sx, sy = self._smooth_pointer(
                                            "gaze", raw[0], raw[1], 6.0, absolute=True)
                                        self.event_ready.emit(InteractionEvent("gaze", "gaze", "move_pointer", 0.82, {
                                            "x": int(sx * (self.width - 1)), "y": int(sy * (self.height - 1))}))
                                if self.head_enabled and not gaze_active:
                                    if self.recenter_requested:
                                        self.head_tracker.reset()
                                        self._forget_source("head")
                                        self.recenter_requested = False
                                    head_feat = self._extract_head_feature(landmarks)
                                    if head_feat is not None:
                                        anchor = None
                                        if self.head_tracker.smoothed is None:
                                            anchor = self._screen_pointer() or self.pointer_last or (0.5, 0.5)
                                        point = self._head_pointer(head_feat, anchor=anchor)
                                        if point is not None and pointer_active:
                                            self.pointer_source = "head"
                                            self.pointer_last = point
                                            self.event_ready.emit(InteractionEvent("head", "head", "move_pointer", 0.75, {
                                                "x": int(point[0] * (self.width - 1)), "y": int(point[1] * (self.height - 1))}))
                        elif now - last_face_seen > 3.0 and now - last_face_warning > 10.0:'''

if old_block in src:
    src = src.replace(old_block, new_block)
    with open('accessflow/inputs/vision.py','w',newline='') as f:
        f.write(src)
    print("replaced")
else:
    print("old block not found")
    # show what we have around there
    idx = src.find('if self.head_enabled and not gaze_active')
    print(repr(src[idx-200:idx+800]))