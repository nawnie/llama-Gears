import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, mock
import tempfile

spec = importlib.util.spec_from_file_location(
    "gears_bench", Path(__file__).resolve().parents[1] / "bench.py"
)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class BenchPortableTests(TestCase):
    def test_profile_validation(self):
        self.assertEqual(bench.parse_profile("moe:20:16")["n_cpu_moe"], 16)
        for value in ["0bad:4:3", "bad:abc:1", "bad:-1:0", "a:1000:0", "bad:1"]:
            with self.assertRaises(Exception):
                bench.parse_profile(value)

    def test_result_parsing_refuses_partial_and_wrong_profiles(self):
        profile = bench.parse_profile("moe:20:12")
        pp = dict(n_gpu_layers=20, n_cpu_moe=12, avg_ts=1000.0, n_prompt=512, n_gen=0)
        tg = dict(n_gpu_layers=20, n_cpu_moe=12, avg_ts=17.0, n_prompt=0, n_gen=128)
        self.assertEqual(len(bench.validate_results([pp, tg], profile, 512, 128)), 2)
        for rows in [[pp], [tg], [dict(pp, n_cpu_moe=0), tg],
                     [dict(pp, avg_ts=float("nan")), tg]]:
            with self.assertRaises(ValueError):
                bench.validate_results(rows, profile, 512, 128)

    def test_subprocess_is_explicit_and_shell_false(self):
        profile = bench.parse_profile("moe:20:12")
        command = bench.build_command(
            Path("bench"), Path("model.gguf"), profile, 512, 128, 3, 8, 0
        )
        fixture = [
            dict(n_gpu_layers=20, n_cpu_moe=12, avg_ts=1000, n_prompt=512, n_gen=0),
            dict(n_gpu_layers=20, n_cpu_moe=12, avg_ts=17, n_prompt=0, n_gen=128),
        ]
        with mock.patch.object(
            bench.subprocess, "run",
            return_value=SimpleNamespace(returncode=0, stdout=json.dumps(fixture), stderr="")
        ) as call:
            self.assertEqual(bench.run_one(command, profile, 10)["status"], "ok")
            self.assertFalse(call.call_args.kwargs["shell"])
        with mock.patch.object(
            bench.subprocess, "run",
            return_value=SimpleNamespace(returncode=0, stdout="bad json", stderr="")
        ):
            self.assertEqual(bench.run_one(command, profile, 10)["status"], "invalid-output")

    def test_dry_run_records_sha_and_does_not_execute(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "test model.gguf"
            model.write_bytes(b"synthetic")
            report = Path(directory) / "results.json"
            with mock.patch.object(bench.platform, "platform", return_value="TestOS"), \
                 mock.patch.object(bench.subprocess, "run") as call:
                code = bench.main([
                    "--bench", "not-installed", "--model", str(model),
                    "--profile", "baseline:0:0", "--hash", "--report", str(report)
                ])
                call.assert_not_called()
            self.assertEqual(code, 0)
            data = json.loads(report.read_text())
            self.assertEqual(data["status"], "dry-run")
            self.assertEqual(data["commands"][0]["argv"][2], str(model))
            self.assertEqual(len(data["model_shards"][0]["sha256"]), 64)
