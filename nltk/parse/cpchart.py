# Natural Language Toolkit: Gaussian Probabilistic Chart Parsers
#
# Copyright (C) 2001-2013 NLTK Project
# Author: Edward Loper <edloper@gradient.cis.upenn.edu>
#         Steven Bird <stevenbird1@gmail.com>
# URL: <http://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Classes and interfaces for associating gaussian continuse probabilities
to the pchart parsing

"""
from __future__ import print_function, unicode_literals

##//////////////////////////////////////////////////////
##  Continuous PCFG Chart Parser
##//////////////////////////////////////////////////////


from nltk.tree import Tree, ProbabilisticTree
from nltk.grammar import Nonterminal, ContinuousWeightedGrammar

from nltk.parse.chart import Chart, LeafEdge, TreeEdge, AbstractChartRule
from nltk.compat import python_2_unicode_compatible

from nltk.parse.pchart import ProbabilisticLeafEdge, InsideChartParser

import numpy as np

# CProbabilistic edges
class CProbabilisticLeafEdge(ProbabilisticLeafEdge):
  # This leaf is the state that can generate gaussian output
    def logprob(self) : return 0.0

class CProbabilisticTreeEdge(TreeEdge):
    def __init__(self, logprob, *args, **kwargs):
        TreeEdge.__init__(self, *args, **kwargs)
        self._logprob = logprob
        # two edges with different probabilities are not equal.
        self._comparison_key = (self._comparison_key, logprob)

    def logprob(self): return self._logprob

    @staticmethod
    def from_production(production, index, p):
        return CProbabilisticTreeEdge(p, (index, index), production.lhs(),
                                     production.rhs(), 0)

# Rules using probabilistic edges
class CProbabilisticBottomUpInitRule(AbstractChartRule):
    NUM_EDGES=0
    def apply_iter(self, chart, grammar):
        for index in range(chart.num_leaves()):
            new_edge = CProbabilisticLeafEdge(chart.leaf(index), index)
            if chart.insert(new_edge, ()):
                yield new_edge

class CProbabilisticBottomUpPredictRule(AbstractChartRule):
    NUM_EDGES=1
    def apply_iter(self, chart, grammar, edge):
        if edge.is_incomplete(): return
        for prod in grammar.productions():
            if edge.lhs() == prod.rhs()[0]:
                new_edge = CProbabilisticTreeEdge.from_production(prod, edge.start(), prod.logprob())
                if chart.insert(new_edge, ()):
                    yield new_edge

class CProbabilisticFundamentalRule(AbstractChartRule):
    NUM_EDGES=2
    def apply_iter(self, chart, grammar, left_edge, right_edge):
        # Make sure the rule is applicable.
        if not (left_edge.end() == right_edge.start() and
                left_edge.nextsym() == right_edge.lhs() and
                left_edge.is_incomplete() and right_edge.is_complete()):
            return

        # Construct the new edge.
        p = left_edge.logprob() + right_edge.logprob()
        new_edge = CProbabilisticTreeEdge(p,
                            span=(left_edge.start(), right_edge.end()),
                            lhs=left_edge.lhs(), rhs=left_edge.rhs(),
                            dot=left_edge.dot()+1)

        # Add it to the chart, with appropriate child pointers.
        changed_chart = False
        for cpl1 in chart.child_pointer_lists(left_edge):
            if chart.insert(new_edge, cpl1+(right_edge,)):
                changed_chart = True

        # If we changed the chart, then generate the edge.
        if changed_chart: yield new_edge

@python_2_unicode_compatible
class CSingleEdgeProbabilisticFundamentalRule(AbstractChartRule):
    NUM_EDGES=1

    _fundamental_rule = CProbabilisticFundamentalRule()

    def apply_iter(self, chart, grammar, edge1):
        fr = self._fundamental_rule
        if edge1.is_incomplete():
            # edge1 = left_edge; edge2 = right_edge
            for edge2 in chart.select(start=edge1.end(), is_complete=True,
                                     lhs=edge1.nextsym()):
                for new_edge in fr.apply_iter(chart, grammar, edge1, edge2):
                    yield new_edge
        else:
            # edge2 = left_edge; edge1 = right_edge
            for edge2 in chart.select(end=edge1.start(), is_complete=False,
                                      nextsym=edge1.lhs()):
                for new_edge in fr.apply_iter(chart, grammar, edge2, edge1):
                    yield new_edge

    def __str__(self):
        return 'Continuous Fundamental Rule'

class CProbabilisticEmissionRule(AbstractChartRule):
  NUM_EDGES = 3 #FIXME the num should be the number of emission states
  def apply_iter(self, chart, grammar, edge):
    if not isinstance(edge, ProbabilisticLeafEdge):
      return
    for state, model in grammar.density().iteritems():
      prob = model(edge.lhs())
      new_edge = CProbabilisticTreeEdge(prob,
                                        (edge.start(), edge.start()),
                                        state, [edge.lhs()], 0)
      if chart.insert(new_edge, ()):
        yield new_edge


class CInsideChartParser(InsideChartParser):
  def sort_queue(self, queue, chart):
    queue.sort(key=lambda edge:edge.logprob())
  def _setlogprob(self, tree, prod_probs, emission):
    if tree.logprob() is not None : return
    lhs = Nonterminal(tree.node)
    rhs = []
    for child in tree:
      if isinstance(child, Tree):
        rhs.append(Nonterminal(child.node))
      else:
        rhs.append(child)
    if isinstance(rhs[0], Nonterminal):
      logprob = prod_probs[lhs, tuple(rhs)]
    else:
      logprob = emission[lhs](rhs[0])

    for child in tree:
      if isinstance(child, Tree):
        self._setlogprob(child, prod_probs, emission)
        logprob += child.logprob()

    tree.set_logprob(logprob)

  def nbest_parse(self, tokens, n=None):
    chart = Chart(list(tokens))
    grammar = self._grammar

    bu_init = CProbabilisticBottomUpInitRule()
    bu = CProbabilisticBottomUpPredictRule()
    fr = CSingleEdgeProbabilisticFundamentalRule()
    em = CProbabilisticEmissionRule()

    queue = []
    for edge in bu_init.apply_iter(chart, grammar):
      if self._trace > 1:
        print('  %-50s [%s]' % (chart.pp_edge(edge,width=2),edge.logprob()))
      queue.append(edge)

    while len(queue) > 0:
      self.sort_queue(queue, chart)
      if self.beam_size:
        self._prune(queue, chart)
      edge = queue.pop()
      if self._trace > 0:
        print('  %-50s [%s]' % (chart.pp_edge(edge,width=2),edge.logprob()))

      queue.extend(em.apply(chart, grammar, edge))
      queue.extend(bu.apply(chart, grammar, edge))
      queue.extend(fr.apply(chart, grammar, edge))

    parses = chart.parses(grammar.start(), ProbabilisticTree)

    prod_probs = {}
    for prod in grammar.productions():
      prod_probs[prod.lhs(), prod.rhs()] = prod.logprob()

    for parse in parses:
      self._setlogprob(parse, prod_probs, grammar.density())

    parses.sort(reverse=True, key=lambda tree: tree.logprob())

    return parses[:n]

class CInsideOutsideChartParser(CInsideChartParser):
  def nbest_parse(self, tokens, n=None):
    chart = Chart(list(tokens))
    grammar = self._grammar
    
    bu_init = CProbabilisticBottomUpInitRule()
    bu = CProbabilisticBottomUpPredictRule()
    fr = CSingleEdgeProbabilisticFundamentalRule()
    em = CProbabilisticEmissionRule()

    queue = []
    for edge in bu_init.apply_iter(chart, grammar):
      if self._trace > 1:
        print('  %-50s [%s]' % (chart.pp_edge(edge,width=2),edge.logprob()))
      queue.append(edge)

    while len(queue) > 0:
      self.sort_queue(queue, chart)
      if self.beam_size:
        self._prune(queue, chart)
      edge = queue.pop()
      if self._trace > 0:
        print('  %-50s [%s]' % (chart.pp_edge(edge,width=2),edge.logprob()))

      queue.extend(em.apply(chart, grammar, edge))
      queue.extend(bu.apply(chart, grammar, edge))
      queue.extend(fr.apply(chart, grammar, edge))

    # construct the pre-requirement
    import pprint
    A = {}
    B = {}
    C = {}
    e = {}
    f = {}
    for prod in grammar.productions() :
      print (prod.lhs(), prod.rhs())
      if len(prod.rhs())> 1 :
        A[(prod.lhs(), prod.rhs()[0], prod.rhs()[1])] = prod.logprob()
      else:
        B[(prod.lhs(), prod.rhs()[0])] = prod.logprob()
    for o in tokens:
      for key, em in grammar.density().iteritems():
        C[(key, o)] = em(o)
    print (A)
    print (B)
    print (C)
    for s, o in enumerate(tokens):
      for key, item in B.iteritems():
        e[(s,s,key[0])] = item+C.get( (key[1],o) )
    for s in range(len(tokens)-1):
      for key, item in A.iteritems():
        print (s, s+1, key, item, e.get((s, s, key[1])), e.get((s+1, s+1, key[2])))
        if (item and e.get((s,s,key[1])) and e.get((s+1,s+1,key[2]))):
          e[(s, s+1, key[0])] = item + e.get((s, s, key[1]))+e.get((s+1, s+1, key[2]))
    
    for dt in range(len(tokens)):
      for s in range(len(tokens)):
        t = s + dt
        if t>s and t<len(tokens):
          for key, item in A.iteritems():
            sumr = 0
            flag = None
            for r in range(s, t):
              print (s, t, r, key, item, e.get((s, r, key[1])), e.get((r+1, t, key[2])) )
              try:
                sumr = sumr + item + e.get((s,r,key[1])) + e.get((r+1, t, key[2]))
                print ("\te(%s,%s,%s)=%s"%(s,t,key[0],sumr ))
                flag = 1
              except TypeError:
                print ("have None ")
                pass
            if flag:
              e[(s,t,key[0])] = sumr
              print ("e(%s,%s,%s)=%s"%(s,t,key[0],sumr))

    pprint.pprint (e)
    pprint.pprint (f)

def demo(choice=None, draw_parses=None, print_parses=None):
    """
    A demonstration of the probabilistic parsers.  The user is
    prompted to select which demo to run, and how many parses should
    be found; and then each parser is run on the same demo, and a
    summary of the results are displayed.
    """
    from nltk.parse import cpchart
    from scipy.stats import norm
    from nltk.grammar import parse_cpcfg
    densityEmission = {
        Nonterminal('s1') : lambda x: norm.logpdf(x, loc=0, scale=1),
        Nonterminal('s2') : lambda x: norm.logpdf(x, loc=20, scale=1),
        #Nonterminal('s3') : lambda x: norm.logpdf(x, loc=-20, scale=1),
    }
    emission = {
        Nonterminal('s1') : lambda x: norm.rvs(loc=0, scale=1, size=x),
        Nonterminal('s2') : lambda x: norm.rvs(loc=20, scale=1, size=x),
        #Nonterminal('s3') : lambda x: norm.rvs(loc=-20, scale=1, size=x),
    }
    g = parse_cpcfg("""
      S0 -> S A [1.0]
      S -> A B [1.0]
      A -> A A [0.5] | s1 [0.5]
      B -> s2 [1.0]
      """, emission, densityEmission)

    #parser = CInsideChartParser(g)
    parser = CInsideOutsideChartParser(g)
    parser.trace(0)
    parses = parser.nbest_parse([0,0,20,0,0], n=1)
    for p in parses:
      print(p)

if __name__ == '__main__':
    demo()
