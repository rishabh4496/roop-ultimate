"""Guards for the canvas preview pipeline, the scrub throttle, and object-URL
lifetime.

Three properties, and none of them is visible to the build, to oxlint, or to
any behavioural test in this suite -- which is why they are asserted against the
source here rather than left to review.

1. THE PREVIEW MUST NOT PUT IMAGE BYTES IN REACT STATE.
   `/api/preview` answers with a base64 data URL. Held in `useState` that is a
   multi-megabyte STRING flowing through a commit of the whole Face Swap panel,
   compared by identity by every consumer, and written to localStorage. It is
   converted to a blob URL at the single point it enters the client. A future
   edit that pipes `res.image` back into state reintroduces the whole problem
   silently -- the UI looks identical.

2. AN OBJECT URL MUST HAVE A REVOKE.
   `URL.createObjectURL` pins its Blob until `revokeObjectURL` is called BY
   NAME. Nothing collects it, no error is raised, and the document here is
   long-lived (Pinokio's webview reload is the only thing that clears it, which
   is why a leak "fixes itself" on a tab switch and so goes unnoticed). Every
   file that creates one must free it.

3. THE SCRUB PATH MUST BE ABORTABLE AND THROTTLED.
   Frame seeks are serialised behind one video decoder on the backend at a flat
   ~125-180 ms each, so an unbounded drag queues decodes whose results are
   discarded on arrival while the frame the user stopped on waits behind them.
   The previous implementation used `new Image()`, which CANNOT be cancelled --
   it could only choose which finished result to ignore.
"""

import os
import re
import unittest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(os.path.dirname(APP), 'react-ui', 'src')


def read(*parts):
    with open(os.path.join(SRC, *parts), encoding='utf-8') as fh:
        return fh.read()


# Comments in this codebase QUOTE the code they replaced ("this used to be
# setSliderPosition(pos)"), which is exactly the string these assertions look
# for. Strip them, or the tests fail on their own documentation.
LINE_COMMENT = re.compile('//.*')
BLOCK_COMMENT = re.compile(r'/\*.*?\*/', re.S)


def code_only(text):
    return LINE_COMMENT.sub('', BLOCK_COMMENT.sub('', text))


def walk_sources():
    for root, _dirs, files in os.walk(SRC):
        for name in files:
            if name.endswith(('.js', '.jsx')):
                path = os.path.join(root, name)
                with open(path, encoding='utf-8') as fh:
                    yield os.path.relpath(path, SRC).replace('\\', '/'), fh.read()


class PreviewBytesNeverEnterReactState(unittest.TestCase):
    def test_preview_response_is_converted_before_setState(self):
        src = read('components', 'FaceSwap.jsx')
        self.assertIn('dataUrlToOwnedBlobUrl(res.image, previewOwner)', src,
                      'the /api/preview response must be converted to a blob URL '
                      'at its single point of entry')
        # The exact regression: handing the data URL straight to state.
        self.assertNotIn('setPreviewSrc(res.image', src)
        self.assertNotIn("image: res.image }", src)

    def test_grid_cells_are_converted_too(self):
        src = read('components', 'faceswap', 'useGridPreviewLoader.js')
        self.assertIn('dataUrlToOwnedBlobUrl(res.image, owner)', src)
        # A grid of six cells held six full-resolution base64 strings in state
        # AND six more in the cache.
        self.assertNotIn('[value]: res.image', src)

    def test_upscale_resends_bytes_not_a_reference(self):
        """`/api/preview_upscale` wants pixels; a blob URL is meaningless to it.

        Sending `previewSrc` verbatim would now post the string "blob:http://..."
        to a backend that runs `_dataurl_to_bgr` on it -- a failure that only
        shows up when someone clicks Upscale.
        """
        src = read('components', 'FaceSwap.jsx')
        self.assertIn('await blobUrlToDataUrl(previewSrc)', src)
        self.assertNotIn('image: previewSrc,', src)


class EveryObjectUrlIsRevoked(unittest.TestCase):
    # objectUrls.js is the registry itself: it creates on behalf of others and
    # frees through releaseOwner/revokeUrl, so a same-file pairing check does
    # not describe it.
    REGISTRY = 'components/faceswap/objectUrls.js'

    def test_creating_file_also_frees(self):
        offenders = []
        for rel, src in walk_sources():
            if rel == self.REGISTRY:
                continue
            if 'URL.createObjectURL' not in src:
                continue
            frees = ('URL.revokeObjectURL' in src
                     or 'revokeUrl' in src
                     or 'releaseOwner' in src)
            if not frees:
                offenders.append(rel)
        self.assertEqual([], offenders,
                         'these files create an object URL and never free one: '
                         + ', '.join(offenders))

    def test_registry_frees_what_it_creates(self):
        src = read('components', 'faceswap', 'objectUrls.js')
        self.assertIn('URL.createObjectURL', src)
        self.assertIn('URL.revokeObjectURL', src)
        # The diagnostic that makes the invariant checkable at runtime rather
        # than only in review.
        self.assertIn('export function __stats', src)

    def test_preview_cache_eviction_revokes(self):
        """The cache is capped at 200 entries. Dropping one without a revoke is
        a permanent leak -- the entry is gone, the bytes are not."""
        src = read('components', 'FaceSwap.jsx')
        cap = src[src.index('const setCachedPreview'):]
        cap = cap[:cap.index('\n  };')]
        self.assertIn('revokeUrl(victim.image)', cap)
        self.assertIn('delete cache[keys[0]]', cap)

    def test_cache_clear_spares_what_is_on_screen(self):
        """A blanket releaseOwner() here would revoke the URL the stage and any
        open comparison grid are still displaying, blanking the picture with no
        error anywhere."""
        src = read('components', 'FaceSwap.jsx')
        block = src[src.index('const clearPreviewCache'):]
        block = block[:block.index('\n  };')]
        self.assertIn('displayed.has(url)', block)
        self.assertNotIn('releaseOwner', block)

    def test_unmount_sweeps_the_owner(self):
        src = read('components', 'FaceSwap.jsx')
        self.assertIn('useEffect(() => () => { releaseOwner(previewOwner); }', src)


class CrossComponentHandoffsCarryBytes(unittest.TestCase):
    """A blob URL is a handle into ONE document's blob store, owned by the
    component that made it. Two consumers outlive that component, so they get
    bytes instead."""

    def test_popout_window_gets_bytes(self):
        src = read('components', 'FaceSwap.jsx')
        block = src[src.index("popoutManager.isOpen()"):]
        block = block[:block.index('}, [previewSrc, rawUrl]);')]
        self.assertIn('blobUrlToDataUrl', block)

    def test_processing_tab_still_gets_bytes(self):
        src = read('components', 'FaceSwap.jsx')
        block = src[src.index('setLastPreview({ rawUrl, frame, maxFrames });'):]
        block = block[:block.index('}, [previewSrc, rawUrl, frame, maxFrames]);')]
        self.assertIn('blobUrlToDataUrl', block)
        note = read('components', 'faceswap', 'lastPreview.js')
        self.assertIn('MUST be self-contained bytes', note)


class ScrubRequestsAreThrottledAndAbortable(unittest.TestCase):
    def test_hook_binds_an_abort_controller(self):
        src = read('components', 'faceswap', 'useThrottledFrameRequest.js')
        self.assertIn('new AbortController()', src)
        self.assertIn('running.ctrl.abort()', src,
                      'a request the pointer has already moved past must be '
                      'cancelled, not merely ignored on arrival')
        self.assertIn('inFlightRef.current?.ctrl.abort()', src)

    def test_throttle_is_150ms(self):
        src = read('components', 'faceswap', 'useThrottledFrameRequest.js')
        self.assertIn('DEFAULT_THROTTLE_MS = 150', src)

    def test_both_seek_paths_use_it(self):
        """The stage frame AND the timeline's hover thumbnail hit the same
        single decoder, so decoration must not be able to delay the frame the
        user is looking at."""
        for rel in (('components', 'FaceSwap.jsx'),
                    ('components', 'faceswap', 'Timeline.jsx')):
            src = read(*rel)
            self.assertIn('useThrottledFrameRequest(', src, '/'.join(rel))
            self.assertIn('throttleMs: 150', src, '/'.join(rel))

    def test_uncancellable_image_loader_is_gone(self):
        self.assertFalse(
            os.path.exists(os.path.join(SRC, 'components', 'faceswap',
                                        'useSequentialImage.js')),
            'useSequentialImage used new Image(), whose load cannot be aborted')
        for rel, src in walk_sources():
            self.assertNotIn('from ./useSequentialImage', src.replace("'", ''), rel)

    def test_worker_aborts_the_fetch_and_the_decode(self):
        src = read('components', 'faceswap', 'decoder.worker.js')
        self.assertIn("type === 'abort'", src)
        self.assertIn('inFlight.get(id)?.abort()', src)
        # Aborting after the bitmap exists must still free it.
        self.assertIn('bitmap.close()', src)

    def test_playback_chunks_are_abortable(self):
        """A playback chunk is up to 48 server-side seeks against the same
        decoder the preview needs."""
        src = read('components', 'faceswap', 'usePlaybackBuffer.js')
        self.assertIn('chunkAborters', src)
        self.assertIn('for (const c of chunkAborters) c.abort();', src)


class TheStageIsAnUncontrolledCanvas(unittest.TestCase):
    def test_canvas_component_exists_and_is_used(self):
        canvas = read('components', 'faceswap', 'PreviewCanvas.jsx')
        self.assertIn('<canvas', canvas)
        self.assertIn('ctx.drawImage', canvas)
        used = read('components', 'faceswap', 'InteractivePreview.jsx')
        self.assertIn('<PreviewCanvas', used)

    def test_no_state_is_set_while_painting(self):
        """The point of the refactor. A setState in the paint or fade path puts
        a commit of the whole panel between the pointer moving and the pixel
        moving."""
        canvas = read('components', 'faceswap', 'PreviewCanvas.jsx')
        paint = canvas[canvas.index('const paint = useCallback'):]
        paint = code_only(paint[:paint.index('const requestPaint')])
        # Any `setSomething(` call would be a state setter reaching React.
        setters = re.findall(r'set[A-Z]\w*\s*\(', paint)
        # ctx.setTransform is the canvas API, not React.
        setters = [x for x in setters if not x.startswith('setTransform')]
        self.assertEqual([], setters, paint)
        # `useState` is not imported at all, so there is nothing to set.
        self.assertNotIn('useState', code_only(canvas))

    def test_decoded_frames_are_closed(self):
        """An ImageBitmap holds memory until close(); a 4K frame is ~33 MB, so
        one per scrub tick outruns any blob-URL leak."""
        canvas = read('components', 'faceswap', 'PreviewCanvas.jsx')
        self.assertIn('releaseOwned', canvas)
        hook = read('components', 'faceswap', 'useThrottledFrameRequest.js')
        self.assertIn('releaseFrame(frameRef.current.img)', hook)

    def test_caller_owned_frames_are_never_closed_by_the_canvas(self):
        """Closing a bitmap someone else still holds is a use-after-free that
        presents as a silently blank stage, not as an error."""
        canvas = read('components', 'faceswap', 'PreviewCanvas.jsx')
        self.assertIn('if (img && !layer.external) releaseFrame(img)', canvas)

    def test_teardown_resets_its_own_bookkeeping(self):
        """Both halves of this were shipped broken and BOTH read as a blank
        stage with no error anywhere.

        React 19 StrictMode mounts, tears down and mounts again. A teardown that
        frees a resource but leaves the flag saying it exists makes the second
        mount a no-op:

          * `rafRef` — `requestPaint` guards on it to avoid stacking frames, so
            an id left set after cancelAnimationFrame makes every later call
            return immediately. The component never paints again, ever.
          * `layer.src` — `loadLayer` returns early when the source has not
            changed, so a src left set means the remount believes the frame is
            already loaded while its bitmap has been released.

        Together they produced a correct-looking <canvas> at its 300x150 default
        size, showing nothing.
        """
        canvas = read('components', 'faceswap', 'PreviewCanvas.jsx')
        teardown = canvas[canvas.index('useEffect(() => () => {'):]
        teardown = teardown[:teardown.index('}, []);')]
        self.assertIn('rafRef.current = 0;', teardown)
        self.assertIn("src: ''", teardown)
        self.assertIn("reportedDimRef.current = ''", teardown)

    def test_the_canvas_takes_the_frames_own_size(self):
        """A <canvas> is a replaced element whose intrinsic size is its
        width/height attributes — 300x150 by default. The stage around it is
        content-sized, so a canvas that is not given the frame's dimensions
        collapses the whole preview to a small box, and the face-box overlay
        (positioned in percentages of this element) stops matching the faces."""
        canvas = read('components', 'faceswap', 'PreviewCanvas.jsx')
        self.assertIn('const sizeToFrame = useCallback', canvas)
        self.assertIn('sizeToFrame(iw, ih);', canvas)
        self.assertIn('MAX_EDGE', canvas)
        # It must be declared before `paint`, which names it in a dep array —
        # a dep array is an argument, evaluated as the body runs.
        self.assertLess(canvas.index('const sizeToFrame = useCallback'),
                        canvas.index('const paint = useCallback'))

    def test_readback_is_not_disabled(self):
        """`desynchronized: true` selects a low-latency path that makes
        getImageData read back blank in Chromium. It buys nothing at video
        rates and it silently disables every pixel check that would reveal this
        component had stopped painting."""
        canvas = read('components', 'faceswap', 'PreviewCanvas.jsx')
        self.assertNotIn('desynchronized: true', code_only(canvas))

    def test_the_canvas_lays_out_like_the_img_it_replaced(self):
        src = read('components', 'faceswap', 'InteractivePreview.jsx')
        for block in src.split('<PreviewCanvas')[1:]:
            head = block[:block.index('/>')]
            self.assertIn('object-contain', head, head[:200])

    def test_the_two_img_layer_crossfade_is_gone(self):
        src = read('components', 'faceswap', 'InteractivePreview.jsx')
        self.assertNotIn('<CrossfadeImage', src)


class CompareSliderIsReusableAndCheap(unittest.TestCase):
    def test_component_exists_and_is_mounted(self):
        slider = read('components', 'faceswap', 'CompareSlider.jsx')
        self.assertIn('const CompareSlider = forwardRef', slider)
        used = read('components', 'faceswap', 'InteractivePreview.jsx')
        self.assertIn('<CompareSlider', used)

    def test_dragging_does_not_move_position_through_state(self):
        """`position` in useState + `clipPath` in a style prop puts a full
        commit between the pointer and the pixel. The position lives in a ref
        and reaches the canvas through its imperative handle."""
        slider = read('components', 'faceswap', 'CompareSlider.jsx')
        block = slider[slider.index('const apply = useCallback'):]
        block = code_only(block[:block.index('// Controlled mode')])
        self.assertIn('posRef.current = pct', block)
        self.assertIn('canvasRef?.current?.setWipe?.(pct)', block)
        self.assertNotIn('setPosition', block)

    def test_it_is_keyboard_reachable(self):
        slider = read('components', 'faceswap', 'CompareSlider.jsx')
        self.assertIn("role=\"slider\"", slider)
        self.assertIn('tabIndex={0}', slider)
        self.assertIn('aria-valuenow', slider)
        self.assertIn('onKeyDown', slider)

    def test_the_announced_value_tracks_the_drag(self):
        """`aria-valuenow` is rendered from a render-time snapshot, and dragging
        deliberately skips the re-render — so it has to be written on the node
        too, or the value a screen reader reads freezes while the curtain
        visibly moves."""
        slider = read('components', 'faceswap', 'CompareSlider.jsx')
        self.assertIn("handle.setAttribute('aria-valuenow'", slider)
        self.assertIn("handle.setAttribute('aria-valuetext'", slider)

    def test_auto_swipe_no_longer_ticks_react_state(self):
        """It ran at 60 Hz. Sixty commits a second of the panel that owns the
        stage, to move a divider."""
        src = read('components', 'faceswap', 'InteractivePreview.jsx')
        block = src[src.index('const autoSwipeRuns'):]
        block = code_only(block[:block.index('// Ends any drag')])
        self.assertIn('compareApiRef.current?.set(100 - pos)', block)
        # One settle on teardown is fine; a per-frame setSliderPosition is not.
        self.assertEqual(1, block.count('setSliderPosition('), block)


class WipeAndDividerAreComplements(unittest.TestCase):
    """`wipe` is how much of the stage the SWAPPED layer covers; the slider's
    own position is the DIVIDER's distance from the left, with the swapped side
    to its right. Mixing them up silently inverts the comparison -- which looks
    plausible, so nothing would report it."""

    def test_slider_mode_inverts(self):
        src = read('components', 'faceswap', 'InteractivePreview.jsx')
        block = src[src.index('const canvasWipe = (() => {'):]
        block = block[:block.index('})();')]
        self.assertIn('return 100 - sliderPosition;', block)
        self.assertIn("if (compareMode === 'blend') return sliderPosition;", block)

    def test_commit_converts_back(self):
        src = read('components', 'faceswap', 'InteractivePreview.jsx')
        self.assertIn('setSliderPosition(100 - pct)', src)
        self.assertIn('defaultPosition={100 - sliderPosition}', src)


if __name__ == '__main__':
    unittest.main()
