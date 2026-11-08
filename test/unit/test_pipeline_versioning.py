"""Collect/resume must not mix old and new system implementations."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from test.evaluation_agent.collect import main, pipeline_fingerprint
from test.unit.test_agent_scoring import gold, output


class PipelineVersionTests(unittest.TestCase):
    def test_runtime_source_change_changes_fingerprint_without_mutating_files(self):
        original_read = Path.read_text
        baseline = pipeline_fingerprint()

        def changed_read(path, *args, **kwargs):
            content = original_read(path, *args, **kwargs)
            return content + "\n# simulated pipeline change\n" if path.name == "generator.py" else content

        with patch.object(Path, "read_text", changed_read):
            self.assertNotEqual(pipeline_fingerprint(), baseline)
        self.assertEqual(pipeline_fingerprint(), baseline)

    def test_collect_new_version_then_resume_skips_completed_rows(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            dataset, target = root / "dataset.jsonl", root / "outputs.jsonl"
            dataset.write_text(json.dumps(gold()) + "\n", encoding="utf-8")
            runners = {v: SimpleNamespace(run=Mock(return_value={**output(), "variant": v}),
                                         diagnostics={"generation_attempts": [], "execution_trace": []})
                       for v in ("baseline", "agent")}
            args = ["--dataset", str(dataset), "--output", str(target)]
            config = SimpleNamespace(as_dict=lambda: {"generator": "offline-stub"})
            with patch("test.evaluation_agent.config.ComparisonConfig.from_env", return_value=config), \
                 patch("test.evaluation_agent.collect.pipeline_fingerprint", return_value="pipeline-v2"), \
                 patch("test.evaluation_agent.runners.build_runners", return_value=runners) as build:
                self.assertEqual(main(args), 0)
                meta = json.loads(target.with_suffix(".run.json").read_text(encoding="utf-8"))
                self.assertEqual(meta["pipeline_sha256"], "pipeline-v2")
                contents = target.read_bytes()
                self.assertEqual(main(args + ["--resume"]), 0)
                self.assertEqual(target.read_bytes(), contents)
                self.assertEqual(build.call_count, 1)
                for runner in runners.values():
                    runner.run.assert_called_once_with(gold()["query"])

    def test_changed_or_legacy_pipeline_metadata_refused_before_model_calls(self):
        for legacy in (False, True):
            with self.subTest(legacy=legacy), TemporaryDirectory() as temp:
                root = Path(temp)
                dataset, target = root / "dataset.jsonl", root / "outputs.jsonl"
                dataset.write_text(json.dumps(gold()) + "\n", encoding="utf-8")
                target.write_text(json.dumps(output()) + "\n", encoding="utf-8")
                from test.evaluation_agent.scoring_data import digest
                config = {"generator": "offline-stub"}
                meta = {"config": config, "dataset_sha256": digest([gold()]), "split": "pilot"}
                if not legacy:
                    meta["pipeline_sha256"] = "old-pipeline"
                metadata_path = target.with_suffix(".run.json")
                metadata_path.write_text(json.dumps(meta), encoding="utf-8")
                originals = target.read_bytes(), metadata_path.read_bytes()
                with patch("test.evaluation_agent.config.ComparisonConfig.from_env",
                           return_value=SimpleNamespace(as_dict=lambda: config)), \
                     patch("test.evaluation_agent.collect.pipeline_fingerprint", return_value="new-pipeline"), \
                     patch("test.evaluation_agent.runners.build_runners") as build, patch("sys.stderr"), \
                     self.assertRaises(SystemExit):
                    main(["--dataset", str(dataset), "--output", str(target), "--resume"])
                build.assert_not_called()
                self.assertEqual(originals, (target.read_bytes(), metadata_path.read_bytes()))


if __name__ == "__main__":
    unittest.main()
