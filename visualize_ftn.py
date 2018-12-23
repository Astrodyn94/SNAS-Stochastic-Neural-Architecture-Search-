from genotypes import PRIMITIVES
from genotypes import Genotype
import torch.nn.functional as F
import sys
import genotypes
from graphviz import Digraph

def genotype(alphas_normal , alphas_reduce):

    def _parse(weights):
      gene = []
      n = 2
      start = 0
      steps = 4
      for i in range(steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2)) # we are going to consider all input nodes
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            #if k != PRIMITIVES.index('none'):
            if k_best is None or W[j][k] > W[j][k_best]:  ###   Choose best operation // We will see...
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene

    steps = 4; multiplier = 4
    concat = range(2 + steps - multiplier, steps+2) ## <==> range(2,6)
    genotype = Genotype(
      normal=_parse(alphas_normal), normal_concat=concat,
      reduce=_parse(alphas_reduce), reduce_concat=concat
    )
    return genotype




def plot(genotype, filename):
  g = Digraph(
      format='pdf',
      edge_attr=dict(fontsize='20', fontname="times"),
      node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5', penwidth='2', fontname="times"),
      engine='dot')
  g.body.extend(['rankdir=LR'])

  g.node("c_{k-2}", fillcolor='darkseagreen2')
  g.node("c_{k-1}", fillcolor='darkseagreen2')
  assert len(genotype) % 2 == 0
  #steps = len(genotype) // 2
  steps = 4 
  for i in range(steps):
    g.node(str(i), fillcolor='lightblue')

  n = 2
  start = 0
  for i in range(steps):
    end = start + n
    #for k in [2*i, 2*i + 1]:
    for k in range(start , end):
      op, j = genotype[k]
      if op != 'none':
        if j == 0:
            u = "c_{k-2}"
        elif j == 1:
            u = "c_{k-1}"
        else:
            u = str(j-2)
        v = str(i)
        g.edge(u, v, label=op, fillcolor="gray")
    n +=1
    start = end
  g.node("c_{k}", fillcolor='palegoldenrod')
  for i in range(steps):
    g.edge(str(i), "c_{k}", fillcolor="gray")

  g.render(filename, view=True)