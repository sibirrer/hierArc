"""
This file manages the workflow in performing a hierarchical analysis. The steps are as follow:

(1) from reading in the constraints from single lenses from a .csv file
(2) process parameter constraints onto a angular diameter posterior likelihood and computing the hyper-parameter
dependence. This process is then saved in a separate file.
(3) feeding in likelihood description of (2) into the hierarchical sampling

In particular, this module ensures that steps (1) - (3) are done self-consistently and allows options for each step.
"""

path2table = ''  # relative path to the table folder that contains lens samples
