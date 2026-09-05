
import threading
import queue
import time


class TTSEngine:
    def __init__(self, engine: str = "pyttsx3", rate: int = 160, volume: float = 1.0):
        self.engine_name = engine
        self._queue = queue.Queue()
        self._speaking = False
        self._last_said = {}          # text -> timestamp, for deduplication
        self.cooldown = 3.0           # seconds before repeating same phrase

        if engine == "pyttsx3":
            self._init_pyttsx3(rate, volume)
        else:
            self._engine = None       # gTTS is stateless, no init needed

        # Background worker thread
        self._worker = threading.Thread(target=self._process_queue, daemon=True)
        self._worker.start()
        print("    TTS engine ready ✓")

    def _init_pyttsx3(self, rate, volume):
        import pyttsx3
        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", rate)
        self._engine.setProperty("volume", volume)
        # Prefer a natural-sounding voice if available
        voices = self._engine.getProperty("voices")
        for v in voices:
            if "en" in v.id.lower():
                self._engine.setProperty("voice", v.id)
                break

    def say(self, text: str, priority: bool = False):
        """
        Queue a speech utterance.
        priority=True skips cooldown check (for urgent hazards).
        """
        if not text:
            return

        now = time.time()
        if not priority:
            last = self._last_said.get(text, 0)
            if now - last < self.cooldown:
                return

        self._last_said[text] = now

        if priority:
            # Clear queue and insert at front for urgent alerts
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self._queue.put((text, True))
        else:
            self._queue.put((text, False))

    def _process_queue(self):
        """Background thread: speak items from queue sequentially."""
        while True:
            try:
                text, _ = self._queue.get(timeout=0.5)
                self._speaking = True
                self._speak(text)
                self._speaking = False
                self._queue.task_done()
            except queue.Empty:
                continue

    def _speak(self, text: str):
        if self.engine_name == "pyttsx3":
            self._engine.say(text)
            self._engine.runAndWait()
        else:
            # gTTS fallback - saves to temp file and plays
            try:
                from gtts import gTTS
                import tempfile
                import os
                import platform
                import subprocess
                tts = gTTS(text=text, lang="en", slow=False)
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                    tts.save(f.name)
                    fname = f.name

                system = platform.system()  # "Windows", "Darwin", or "Linux"
                if system == "Windows":
                    # Play and block until finished using the playsound package
                    # (pip install playsound==1.2.2)
                    try:
                        from playsound import playsound
                        playsound(fname)
                    except ImportError:
                        # Fallback: open with default associated app (non-blocking)
                        os.startfile(fname)
                        time.sleep(2)  # give it a moment before we delete the file
                elif system == "Darwin":
                    subprocess.run(["afplay", fname], check=False)
                else:
                    subprocess.run(["mpg123", "-q", fname], check=False)

                try:
                    os.unlink(fname)
                except OSError:
                    pass  # file may still be locked by the player on some systems
            except Exception as e:
                print(f"[TTS gTTS error] {e}")

    def is_speaking(self):
        return self._speaking
