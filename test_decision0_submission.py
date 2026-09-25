"""Transport/boundary tests, not model-quality measurements."""
import copy
import hashlib
import json
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import decision0_probability_submission as s
import contrast_v7 as original


def request():
    return {'id': 'fixture', 'state': 'The color is blue.',
            'question': {'type': 'choice', 'instructions': 'Choose the named color.',
                         'criteria': {'blue': 'Blue', 'amber': 'Amber'}},
            'labels': ['blue', 'amber']}


class BoundaryTests(unittest.TestCase):
    def test_frozen_sources(self):
        s.verify_core()

    def test_canonical_original(self):
        self.assertEqual(s.canonical_request(request()), original.canonical(request()))

    def test_label_order_not_sortable_map(self):
        self.assertEqual(s.canonical_request(request())['labels'], ['blue', 'amber'])
        self.assertEqual(list(s.canonical_request(request())['question']['criteria']), ['amber', 'blue'])

    def test_no_input_mutation(self):
        x = request(); before = copy.deepcopy(x)
        s.canonical_request(x)
        self.assertEqual(x, before)

    def test_exact_request_rejects_gold(self):
        x = request(); x['expected'] = 'blue'
        with self.assertRaises(ValueError):
            s.canonical_request(x)

    def test_evaluator_metadata_stripped(self):
        x = SimpleNamespace(**request(), expected='DO_NOT_INFER', provenance={'key': 'SECRET'})
        self.assertEqual(s.task_payload(x), s.canonical_request(request()))
        x.expected = 'OTHER'
        self.assertEqual(s.task_payload(x), s.canonical_request(request()))

    def test_http_metadata_stripped(self):
        x = dict(request(), expected=None, provenance={}, group='hidden-group')
        self.assertEqual(s.task_payload(x), s.canonical_request(request()))

    def test_id_not_in_prompt(self):
        x = request(); x['id'] = 'NOT_IN_MODEL_INPUT'
        self.assertNotIn(x['id'], json.dumps(original.e4.messages(s.canonical_request(x), 'reason')))

    def test_duplicate_labels(self):
        x = request(); x['labels'] = ['blue', 'blue']
        with self.assertRaises(ValueError):
            s.canonical_request(x)

    def test_newline_label(self):
        x = request(); x['labels'] = ['a\nb', 'blue']
        with self.assertRaises(ValueError):
            s.canonical_request(x)

    def test_too_many_labels(self):
        x = request(); x['labels'] = [str(i) for i in range(17)]
        with self.assertRaises(ValueError):
            s.canonical_request(x)

    def test_score(self):
        x = request(); x['question']['type'] = 'score'; x['question']['criteria'] = ['low', 'high']
        x['labels'] = ['0', '1']
        self.assertEqual(s.canonical_request(x)['labels'], ['0', '1'])

    def test_score_reversed_invalid(self):
        x = request(); x['question']['type'] = 'score'; x['labels'] = ['1', '0']
        with self.assertRaises(ValueError):
            s.canonical_request(x)

    def test_noul(self):
        x = request(); x['question']['type'] = 'noul'; x['labels'] = ['no', 'yes']
        self.assertEqual(s.canonical_request(x)['labels'], ['no', 'yes'])

    def test_nonfinite_input(self):
        x = request(); x['state'] = float('nan')
        with self.assertRaises(ValueError):
            s.canonical_request(x)

    def test_stable_probabilities(self):
        p = s.calibrated_vector([10000., 10001.])
        self.assertAlmostEqual(sum(p), 1.)
        self.assertGreater(p[1], p[0])

    def test_nonfinite_logits(self):
        for x in (float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                s.calibrated_vector([x, 1.])

    def test_distribution_matches_v13_math(self):
        z = [-1.8, 0.0, 2.4]
        v = [math.exp((x-max(z))/1.0218971486541166) for x in z]
        self.assertEqual(s.calibrated_vector(z), [x/sum(v) for x in v])

    def fake_infer(self, logits):
        trace = {'text': 'FINAL: blue', 'input_tokens': 20, 'output_tokens': 5}
        obs = {'logits': logits, 'input_tokens': 25, 'prompt_sha256': 'fixed'}
        with patch.object(original, 'generate', return_value=trace) as generation, \
             patch.object(original.h5, 'readout', return_value=obs) as readout:
            result = s.infer(object(), request())
            generation.assert_called_once()
            readout.assert_called_once()
            self.assertEqual(generation.call_args.args[2], 480)
            self.assertEqual(readout.call_args.args[2], trace['text'])
            return result

    def test_answer_comes_from_readout_not_final(self):
        result = self.fake_infer([-2., 3.])
        self.assertEqual(result['label'], 'amber')

    def test_lexicographic_tie(self):
        self.assertEqual(self.fake_infer([1., 1.])['label'], 'amber')

    def test_full_token_accounting(self):
        result = self.fake_infer([2., 1.])
        self.assertEqual(result['usage']['prompt_tokens'], 45)
        self.assertEqual(result['usage']['completion_tokens'], 5)
        self.assertEqual(result['usage']['total_tokens'], 50)

    def test_no_prompt_or_draft_text_in_receipt(self):
        result = self.fake_infer([2., 1.])
        self.assertNotIn('FINAL: blue', json.dumps(result))
        self.assertNotIn('The color is blue.', json.dumps(result))

    def test_missing_vector_rejected(self):
        with self.assertRaises(ValueError):
            self.fake_infer([1.])

    def test_no_tariff_fabrication(self):
        adapter = s.Decision0Adapter(runtime=object())
        self.assertIsNone(adapter.price_input_per_m)
        self.assertIsNone(adapter.price_output_per_m)


if __name__ == '__main__':
    unittest.main()
