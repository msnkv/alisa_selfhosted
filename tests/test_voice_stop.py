"""Стоп-команда, эталон воспроизведения и отказ без сырого ASR."""

import json
import queue
import threading
import unittest
import sys
import time
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np

from assistant.config import VoiceStopConfig
from assistant.voice_stop import FRAME, RATE, StopDetector, VoiceStop, find_stop, resample


def result(text="стоп", confidence=0.95, duration=0.3):
    return {"text": text, "result": [
        {"word": word, "conf": confidence, "start": 0, "end": duration}
        for word in text.split()
    ]}


def recognizer(answer=None):
    return NS(Reset=Mock(), AcceptWaveform=Mock(return_value=True),
              Result=Mock(return_value=json.dumps(result() if answer is None else answer)))


class StopDetectorTests(unittest.TestCase):
    def setUp(self):
        self.near = np.full(FRAME, 0.08, dtype=np.float32)
        self.zero = np.zeros(FRAME, dtype=np.float32)

    def detector(self, answer=None):
        rec = recognizer(answer)
        detector = StopDetector(NS(process=lambda near, far: near), rec, Mock(return_value=True))
        for _ in range(30):
            detector.audio.append(np.full(160, 0.08, dtype=np.float32))
        detector.samples_seen = 30 * 160
        return detector, rec

    def test_any_stop_word_goes_to_verification(self):
        # «Стоп, стоп» и «так, стоп» — это команда. Раньше первый проход
        # требовал ровно одно слово и отбрасывал именно повтор после
        # промаха. Похожие слова теперь отсекает только проверка.
        for answer in (result(), result("стоп стоп"), result("так стоп"), result("[unk] стоп [unk]")):
            with self.subTest(answer=answer):
                detector, _ = self.detector(answer)
                self.assertTrue(detector.process(self.near, self.zero))
                detector.verify.assert_called_once()

    def test_first_pass_without_plausible_stop_word_is_not_verified(self):
        for answer in (result("остановись"), result("[unk]"), result(confidence=0.5),
                       result(duration=0.05), result(duration=3), {"text": "стоп"}):
            with self.subTest(answer=answer):
                detector, _ = self.detector(answer)
                self.assertFalse(detector.process(self.near, self.zero))
                detector.verify.assert_not_called()

    def test_second_stop_word_gets_its_own_chance(self):
        detector, _ = self.detector(result("стоп стоп"))
        detector.verify.side_effect = [False, True]
        self.assertTrue(detector.process(self.near, self.zero))
        self.assertEqual(detector.verify.call_count, 2)

    def test_rejections_are_counted_for_the_summary(self):
        detector, _ = self.detector()
        detector.verify.return_value = False
        detector.verify.reason = ("нет слова", "«топ»")
        with self.assertLogs("assistant.voice_stop", level="INFO") as logs:
            detector.process(self.near, self.zero)
        self.assertIn("проверка нет слова «топ»", logs.output[0])
        self.assertEqual(detector.tally(), "кандидатов «стоп» 1, принято 0, отказы: проверка 1")

    def test_verification_wants_the_word_stop_itself(self):
        for answer, expected in ((result(), True), (result("так стоп"), True),
                                 (result("топ"), False), (result("стопка"), False),
                                 (result("сто пудов"), False), (result(confidence=0.75), False)):
            with self.subTest(answer=answer):
                self.assertEqual(find_stop(answer, 0.8)[0] is not None, expected)

    def test_partial_result_without_word_confidence_cannot_stop(self):
        detector, rec = self.detector()
        rec.AcceptWaveform.return_value = False
        self.assertFalse(detector.process(self.near, self.zero))
        rec.Result.assert_not_called()

    def test_blocked_echo_is_never_fed_to_recognizer(self):
        detector, rec = self.detector()
        for _ in range(150):
            self.assertFalse(detector.process(self.near, self.near, blocked=True, playing=True))
        rec.AcceptWaveform.assert_not_called()
        rec.AcceptWaveform.return_value = False
        for _ in range(30):
            detector.process(self.near, self.zero)
        final = result()
        final["result"][0].update(start=(detector.samples_seen - 4800) / 16000,
                                   end=detector.samples_seen / 16000)
        rec.Result.return_value = json.dumps(final)
        rec.AcceptWaveform.return_value = True
        self.assertTrue(detector.process(self.near, self.zero))

    def test_aec_warmup_does_not_pass_initial_echo_to_asr(self):
        detector, rec = self.detector()
        for _ in range(79):
            self.assertFalse(detector.process(self.near, self.near, playing=True))
        rec.AcceptWaveform.assert_not_called()

    def test_quiet_aec_residual_cannot_trigger_stop(self):
        detector, _ = self.detector()
        self.assertFalse(detector.process(self.near * 0.001, self.zero))

    def test_aec_failure_never_falls_back_to_raw_audio(self):
        rec = recognizer()
        for processor in (NS(process=Mock(side_effect=RuntimeError("DSP failed"))),
                          NS(process=lambda *args: np.full(FRAME, np.nan, dtype=np.float32))):
            with self.subTest(processor=processor):
                with self.assertRaises((RuntimeError, ValueError)):
                    StopDetector(processor, rec, Mock(return_value=True)).process(self.near, self.zero)
        rec.AcceptWaveform.assert_not_called()

    def test_pcm_is_16khz_after_aec_and_decimation(self):
        detector, rec = self.detector()
        detector.process(self.near, self.zero)
        self.assertEqual(len(rec.AcceptWaveform.call_args.args[0]), 160 * 2)

    def test_unsafe_confidence_setting_is_rejected(self):
        with self.assertRaises(ValueError):
            StopDetector(None, None, None, 0.1)

    def test_second_pass_rejects_false_keyword_candidate(self):
        detector, _ = self.detector()
        detector.verify.return_value = False
        self.assertFalse(detector.process(self.near, self.zero))
        detector.verify.assert_called_once()

    def test_second_pass_never_runs_without_successful_aec_candidate(self):
        detector, _ = self.detector(result("стол"))
        self.assertFalse(detector.process(self.near, self.zero))
        detector.verify.assert_not_called()

    def test_vosk_reset_preserves_audio_timeline_for_second_pass(self):
        detector, _ = self.detector()
        count = detector.samples_seen
        detector.reset()
        self.assertEqual(detector.samples_seen, count)
        self.assertEqual(len(detector.audio), 0)


class VoiceStopAudioTests(unittest.TestCase):
    def controller(self):
        controller = VoiceStop(None, VoiceStopConfig())
        controller.cancelled = threading.Event()
        return controller

    def test_callback_reference_is_exactly_what_goes_to_speaker(self):
        controller = self.controller()
        controller.samples = np.linspace(-0.1, 0.1, FRAME, dtype=np.float32)
        output = np.zeros((FRAME, 1), dtype=np.float32)
        near = np.ones_like(output) * 0.01
        controller.callback(near, output, FRAME, NS(outputBufferDacTime=1.2, currentTime=1.0), False)
        observed, reference, blocked, playing = controller.frames.get_nowait()
        np.testing.assert_array_equal(reference, output[:, 0])
        np.testing.assert_array_equal(observed, near[:, 0])
        self.assertTrue(playing)
        self.assertFalse(blocked)
        self.assertAlmostEqual(controller.dac_end, 1.21)

    def test_own_stop_word_and_echo_tail_are_blocked(self):
        controller = self.controller()
        controller.samples = np.ones(FRAME, dtype=np.float32) * 0.05
        controller.own_stop = True
        output = np.zeros((FRAME, 1), dtype=np.float32)
        with patch("assistant.voice_stop.time.monotonic", return_value=10):
            controller.callback(output.copy(), output, FRAME, NS(outputBufferDacTime=1.2, currentTime=1.0), False)
        self.assertTrue(controller.frames.get_nowait()[2])
        with patch("assistant.voice_stop.time.monotonic", return_value=10.6):
            controller.callback(output.copy(), output, FRAME, NS(outputBufferDacTime=1.8, currentTime=1.6), False)
        self.assertTrue(controller.frames.get_nowait()[2])
        with patch("assistant.voice_stop.time.monotonic", return_value=10.8):
            controller.callback(output.copy(), output, FRAME, NS(outputBufferDacTime=2.0, currentTime=1.8), False)
        self.assertFalse(controller.frames.get_nowait()[2])

    def test_queue_overflow_disables_recognition_without_muting_reply(self):
        controller = self.controller()
        controller.frames = queue.Queue(maxsize=1)
        controller.frames.put(None)
        controller.samples = np.full(FRAME, 0.05, dtype=np.float32)
        output = np.zeros((FRAME, 1), dtype=np.float32)
        with self.assertLogs("assistant.voice_stop", level="WARNING"):
            controller.callback(output.copy(), output, FRAME, NS(outputBufferDacTime=0, currentTime=0), False)
        self.assertTrue(controller.failed.is_set())
        self.assertTrue(np.any(output))

    def test_queue_holds_seconds_not_half_a_second(self):
        controller = self.controller()
        self.assertGreaterEqual(controller.frames.maxsize * FRAME / RATE, 3)
        controller.processed, controller.busy_s, controller.worst_s, controller.deepest = 100, 0.05, 0.02, 7
        self.assertIn("очередь до 7 из", controller.summary())

    def test_audio_discontinuity_disables_voice_cancellation(self):
        controller = self.controller()
        output = np.zeros((FRAME, 1), dtype=np.float32)
        with self.assertLogs("assistant.voice_stop", level="WARNING"):
            controller.callback(output.copy(), output, FRAME, NS(outputBufferDacTime=0, currentTime=0), True)
        self.assertTrue(controller.failed.is_set())
        self.assertTrue(controller.frames.empty())

    def test_resampling_preserves_default_voice_rate(self):
        samples = np.linspace(-0.1, 0.1, 480, dtype=np.float32)
        self.assertIs(resample(samples, 48000, 48000), samples)
        self.assertEqual(len(resample(samples, 16000, 48000)), 1440)

    def test_stop_does_not_wait_for_entire_playback(self):
        controller = self.controller()
        controller.stream = NS(abort=Mock())
        controller.stop_audio()
        controller.stream.abort.assert_called_once()
        self.assertTrue(controller.done.is_set())

    def test_play_waits_for_dac_not_only_last_queued_frame(self):
        controller = self.controller()
        beginning = time.monotonic()
        class Stream:
            active = True
            @property
            def time(self):
                return time.monotonic() - beginning
        controller.stream = Stream()
        finished = threading.Event()
        worker = threading.Thread(target=lambda: (controller.play(np.ones(FRAME, dtype=np.float32), RATE), finished.set()))
        worker.start()
        deadline = time.monotonic() + 1
        while len(controller.samples) == 0 and time.monotonic() < deadline:
            time.sleep(0.001)
        output = np.zeros((FRAME, 1), dtype=np.float32)
        controller.callback(output.copy(), output, FRAME, NS(outputBufferDacTime=0.1, currentTime=0), False)
        self.assertFalse(finished.wait(0.02), "хвост голоса мог быть обрезан до DAC")
        worker.join(1)
        self.assertTrue(finished.is_set())

    def test_processing_error_disables_listener_but_keeps_audio_stream(self):
        controller = self.controller()
        stream = NS(start=Mock(), abort=Mock(), close=Mock())
        rec = recognizer()
        rec.SetWords = Mock()
        modules = {
            "sounddevice": NS(Stream=Mock(return_value=stream)),
            "pywebrtc_audio": NS(AudioProcessor=Mock(return_value=NS(process=Mock(side_effect=RuntimeError("AEC failed"))))),
            "vosk": NS(KaldiRecognizer=Mock(return_value=rec)),
        }
        with patch.dict(sys.modules, modules):
            with self.assertLogs("assistant.voice_stop", level="WARNING"):
                controller.start(controller.cancelled)
                controller.frames.put((np.ones(FRAME,dtype=np.float32),np.zeros(FRAME,dtype=np.float32),False,False))
                self.assertTrue(controller.failed.wait(1))
            rec.AcceptWaveform.assert_not_called()
            stream.abort.assert_not_called()
            controller.close()
        stream.close.assert_called_once()


class RealEchoTests(unittest.TestCase):
    def test_actual_webrtc_suppresses_delayed_reverberant_reference(self):
        try:
            from pywebrtc_audio import AudioProcessor
        except ImportError:
            self.skipTest("установи pywebrtc-audio для проверки настоящего AEC")
        rng = np.random.default_rng(3)
        size = RATE * 4
        far = np.convolve(rng.normal(0, 0.15, size), np.ones(7) / 7, mode="same").astype(np.float32)
        near = np.zeros_like(far)
        near[3360:] += 0.5 * far[:-3360]
        near[4320:] += 0.12 * far[:-4320]
        processor = AudioProcessor(sample_rate=RATE, echo_cancellation=True, noise_suppression=True)
        cleaned = np.concatenate([processor.process(near[i:i+FRAME], far[i:i+FRAME]) for i in range(0, size, FRAME)])
        reduction = 10 * np.log10(np.mean(near[-RATE:] ** 2) / max(1e-15, np.mean(cleaned[-RATE:] ** 2)))
        self.assertGreater(reduction, 15, f"эхо ослаблено только на {reduction:.1f} дБ")


if __name__ == "__main__":
    unittest.main()
