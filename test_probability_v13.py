import copy, json, math, random, unittest
from unittest.mock import Mock, patch
import probability_v13 as p
import probability_boundary as b

def req():
    return {"id":"t","state":"An object is red.","question":{"type":"choice",
        "instructions":"Choose color.","criteria":{"red":"red","blue":"blue"}},"labels":["red","blue"]}

class BoundaryTests(unittest.TestCase):
    def test_uniform(self): self.assertEqual(b.probabilities([0.,0.]),[.5,.5])
    def test_shift_invariance(self):
        for a,z in zip(b.probabilities([3,4,5]),b.probabilities([10003,10004,10005])):
            self.assertAlmostEqual(a,z,13)
    def test_probability_sum(self):
        for n in range(2,17): self.assertAlmostEqual(sum(b.probabilities(list(range(n)))),1)
    def test_nonfinite(self):
        for v in (float("nan"),float("inf"),-float("inf")):
            with self.assertRaises(ValueError): b.probabilities([v,0])
    def test_temperature(self):
        for t in (0,-1,float("nan"),float("inf")):
            with self.assertRaises(ValueError): b.probabilities([0,1],t)
    def test_argmax_preserved(self):
        for t in (.25,1,8):
            self.assertEqual(b.render(req(),[0,2],b.Calibration(t))["answer"],"blue")
    def test_no_label_distribution(self):
        with self.assertRaises((TypeError,ValueError)): b.render(req(),"red")
    def test_vector_length(self):
        with self.assertRaises(ValueError): b.render(req(),[0])
    def test_score_expectation(self):
        r=req();r["question"]["type"]="score";r["labels"]=["0","1","2"]
        self.assertEqual(b.render(r,[0,0,0])["expected_level"],1)
    def test_score_index_validation(self):
        r=req();r["question"]["type"]="score"
        with self.assertRaises(ValueError): b.render(r,[0,1])
    def test_noul(self):
        r=req();r["question"]["type"]="noul";r["labels"]=["yes","no"]
        self.assertEqual(b.render(r,[0,0])["value"],.5)
    def test_reject_targets(self):
        r=req();r["expected"]="red"
        with self.assertRaises(ValueError): b.render(r,[0,0])
    def test_duplicate_labels(self):
        r=req();r["labels"]=["red","red"]
        with self.assertRaises(ValueError): b.render(r,[0,0])
    def test_eval_rejected_from_fit(self):
        with self.assertRaises(ValueError):
            b.fit_temperature([{"partition":"jevbench","target":[0,1],"logits":[0,1]}],[1])
    def test_soft_target_not_collapsed(self):
        rows=[{"partition":"calibration_fit","target":[.5,.5],"logits":[0,2]}]
        self.assertEqual(b.fit_temperature(rows,[1,2,4])["temperature"],4)
    def test_fit_argmax_unchanged(self):
        rows=[{"partition":"calibration_fit","target":[0,1],"logits":[0,2]}]
        self.assertEqual(b.fit_temperature(rows,[.5,1,2])["temperature"],.5)
    def test_output_consistent(self):
        out=b.render(req(),[2,1]); self.assertEqual(max(out["probabilities"],key=out["probabilities"].get),out["answer"])
    def test_concentration_not_correctness(self):
        out=b.render(req(),[0,0]);self.assertEqual(out["distribution_concentration"],0)
        self.assertFalse(out["semantic_correctness_verified"])

class InferenceTests(unittest.TestCase):
    def test_mask_final_only(self):
        self.assertEqual(p.masked("Calculation 2+2=4.\nFINAL: red"),"Calculation 2+2=4.\n")
    def test_mask_multiple_and_whitespace(self):
        self.assertEqual(p.masked(" FINAL: a\nx\n\tFINAL: b\n"),"x\n")
    def test_mask_no_change(self):
        self.assertEqual(p.masked("The phrase FINAL: red is discussed."),"The phrase FINAL: red is discussed.")
    def test_mask_empty(self): self.assertEqual(p.masked("FINAL: red"),"")
    def test_soft_public_target(self):
        r=req();r.update(expected="red",provenance={"gold_probs":{"red":.7,"blue":.3}})
        self.assertEqual(p.target(r),[.7,.3])
    def test_hard_target(self):
        r=req();r["expected"]="blue";self.assertEqual(p.target(r),[0,1])
    def test_source_target(self):
        r=req();r.update(expected="red",target_probs=[.4,.6]);self.assertEqual(p.target(r),[.4,.6])
    def test_invalid_target(self):
        r=req();r["target_probs"]=[.1,.1]
        with self.assertRaises(ValueError):p.target(r)
    def test_input_omits_key(self):
        r=req();r["expected"]="red";self.assertNotIn("expected",p.request(r))
    def test_unconditional_readout(self):
        rt=Mock();rt.score.return_value=({"logits":[1,0]},None)
        trace={"text":"FINAL: red","output_tokens":3}
        with patch.object(p.old.h5,"readout",return_value={"logits":[0,1]}) as read,patch.object(p.old,"generate") as gen:
            out=p.solve(rt,req(),trace)
            self.assertEqual(read.call_count,2);gen.assert_not_called()
            self.assertEqual(out["readouts"]["full"]["logits"],[0,1])
    def test_live_generation(self):
        rt=Mock();rt.score.return_value=({"logits":[1,0]},None)
        with patch.object(p.old.h5,"readout",return_value={"logits":[0,1]}),patch.object(p.old,"generate",return_value={"text":"FINAL: red","output_tokens":3}) as gen:
            out=p.solve(rt,req())
            gen.assert_called_once()
            self.assertEqual(gen.call_args.args[-1],480)
            self.assertFalse(out["trace_reused"])
    def test_no_silent_readout_failure(self):
        rt=Mock();rt.score.return_value=({"logits":[1,0]},None)
        with patch.object(p.old.h5,"readout",return_value={"logits":[float("nan"),1]}),self.assertRaises(ValueError):
            p.solve(rt,req(),{"text":"a","output_tokens":1})
    def test_request_not_mutated(self):
        r=req();s=copy.deepcopy(r);p.request(r);self.assertEqual(r,s)
    def test_temperature_grid_contains_identity(self):
        self.assertIn(1.0,p.TEMPERATURES);self.assertEqual(len(p.TEMPERATURES),161)
    def test_temperature_grid_bounds(self):
        self.assertEqual((min(p.TEMPERATURES),max(p.TEMPERATURES)),(.25,8.))
    def test_exact_reference_logloss(self):
        self.assertAlmostEqual(-sum(.5*v for v in b.log_probabilities([0,0])),math.log(2))

class AddedBoundaryTests(unittest.TestCase):
    def test_lexical_ties(self):
        r=req();r["labels"]=["z","a"]
        self.assertEqual(b.render(r,[0.,0.])["answer"],"a")
    def test_large_shift_log_probability(self):
        for a,z in zip(b.log_probabilities([1.,2.],.25),b.log_probabilities([100001.,100002.],.25)):
            self.assertAlmostEqual(a,z,13)
    def test_impossible_numeric_range_rejected(self):
        with self.assertRaises(ValueError):b.log_probabilities([-1e308,1e308],.25)

if __name__=="__main__":unittest.main()
