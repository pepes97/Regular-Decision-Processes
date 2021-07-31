# Introduction

Regular Decision Processes (RDPs) have
been recently introduced as a non-Markovian extension of
MDPs that does not require knowing or hypothesizing a hidden
state. An RDP is a fully observable, non-Markovian model in
which the next state and reward are a stochastic function of the
entire history of the system. However, this dependence on the
past is restricted to regular functions. That is, the next state
distribution and reward depend on which regular expression
the history satisfies. An RDP can be transformed into an MDP by extending
the state of the RDP with variables that track the satisfaction of the regular expression governing the RDP dynamics.
Thus, essentially, to learn an RDP, we need to learn these
regular expressions. 


The works of Abadi and Brafman [(Abadi and Brafman, 2020)](https://www.ijcai.org/proceedings/2020/270) and Ronca and De Giacomo [(Ronca and De Giacomo, 2021)](https://arxiv.org/abs/2105.06784) provide two significant contributions to the analysis and solution of RDPs.  
The first work focuses on the use of a deterministic Mealy Machine to specify the RDP where, for each state-action pair, a cluster is emitted. Each cluster has an associated distribution over the system states and reward signal. Then, they formulate the first algorithm to learn RDPs from data that merges histories with similar distributions to generate the best model from which the Mealy Machine is produced.
Finally, the latter is exploited to provide the distributions needed for running MCTS [(Kocsis and Szepesvári, 2006)](https://link.springer.com/chapter/10.1007/11871842_29), [(David et all., 2010)](https://dspace.mit.edu/handle/1721.1/100395).
Four domains (i.e., `RotatingMAB`, `CheatMAB`, `MalfunctionMAB` and `RotatingMaze`) are used for the evaluation.

On the other hand, the second work shows an algorithm that learns a probabilistic deterministic finite automaton (PDFA) that represents RDP. Then, it computes an intermediate policy by solving the MDP obtained from the PDFA. From such policy, the final one is derived by composing it with the transition function of the PDFA. The whole process, repeated multiple times, ensures to reach an $\epsilon$-optimal policy in polynomial time. 

Our project consists of the implementation of the algorithm provided by [(Abadi and Brafman, 2020)](https://www.ijcai.org/proceedings/2020/270) without considering propositions while trying to reproduce the same results as the authors.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/pepes97/Regular-Decision-Processes/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
