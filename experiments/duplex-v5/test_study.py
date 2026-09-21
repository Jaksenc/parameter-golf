import unittest
from fractions import Fraction
from study import *

class TestStudy(unittest.TestCase):
 def bank(self):return number_bank('Sales rose from 20 to 25. Margin is 12.5%. Debt was (300).','What changed?')
 def test_decimal(self):self.assertEqual(parse_number('0.1')+parse_number('0.2'),Fraction(3,10))
 def test_percent(self):self.assertEqual(parse_number('12.5%'),Fraction(1,8))
 def test_accounting_negative(self):self.assertEqual(self.bank()['N3']['value'],'-300')
 def test_bound_sources(self):
  text='Sales 20, margin 5%.';bank=number_bank(text,'Year 2001?')
  for v in bank.values():self.assertEqual((text if v['source']=='evidence' else 'Year 2001?')[v['start']:v['end']],v['text'])
 def test_exact_program(self):self.assertEqual(program_value('PROGRAM=(N1-N0)/N0',self.bank())[0],Fraction(1,4))
 def test_literal_invention(self):
  with self.assertRaises(ValueError):program_value('N0+99',self.bank())
 def test_literal_only(self):
  with self.assertRaises(ValueError):program_value('100/1',self.bank())
 def test_reject_call(self):
  with self.assertRaises(ValueError):program_value('sum([N0,N1])',self.bank())
 def test_reject_attribute(self):
  with self.assertRaises(ValueError):program_value('N0.denominator',self.bank())
 def test_reject_unknown(self):
  with self.assertRaises(ValueError):program_value('N55+N0',self.bank())
 def test_reject_exp(self):
  with self.assertRaises(ValueError):program_value('N0**100',self.bank())
 def test_reject_bool(self):
  with self.assertRaises(ValueError):program_value('N0+True',self.bank())
 def test_zero_division(self):
  with self.assertRaises(ZeroDivisionError):program_value('N0/0',self.bank())
 def test_reference(self):self.assertEqual(ref_program('subtract(25, 20), divide(#0, 20)')[0],Fraction(1,4))
 def test_bad_reference(self):
  with self.assertRaises(ValueError):ref_program('table_sum(foo, none)')
 def test_reference_forward(self):
  with self.assertRaises(ValueError):ref_program('add(#0, 1)')
 def test_final(self):self.assertEqual(final_number('A short calculation.\nFINAL=25%'),Fraction(1,4))
 def test_multiple_final(self):
  with self.assertRaises(ValueError):final_number('FINAL=1\nFINAL=2')
 def test_missing_final(self):
  with self.assertRaises(ValueError):final_number('The answer is 10')
 def test_gold_input_isolation(self):
  raw={'pre_text':['Source information.'],'post_text':[],'table':[['A','20']],'model_input':'SECRET','qa':{'program':'SECRET','answer':'SECRET','gold_inds':'SECRET'}}
  self.assertNotIn('SECRET',source_state(raw))
 def test_prompt_isolation(self):
  inp={'id':'test','family':'finqa','split':'evaluation','evidence':'There are 2 tokens.','question':'How many?','bank':number_bank('There are 2 tokens.','How many?')}
  for a in ARMS:
   msg=task_messages(inp,a);self.assertNotIn('evaluation',str(msg));self.assertEqual(len(msg),2)
  inp['gold']='leak'
  with self.assertRaises(AssertionError):task_messages(inp,'direct')
if __name__=='__main__':unittest.main()
