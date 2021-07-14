import subprocess
import graphviz
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from IPython.display import display

def flexfringe(*args, **kwargs):
  """Wrapper to call the flexfringe binary

   Keyword arguments:
   position 0 -- input file with trace samples
   kwargs -- list of key=value arguments to pass as command line arguments
  """  
  command = ["--help"]

  if(len(kwargs) >= 1):
    command = []
    for key in kwargs:
      command += ["--" + key + "=" + kwargs[key]]

  result = subprocess.run(["../../flexfringe/dfasat/flexfringe",] + command + [args[0]], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
  print(result.returncode, result.stdout, result.stderr)

  
  try:
    with open("prova.txt.dat.ff.final.dot") as fh:
      return fh.read()
  except FileNotFoundError:
    pass
  
  return "No output file was generated."

def show(data):
  """Show a dot string as (inline-) PNG

    Keyword arguments:
    data -- string formated in graphviz dot language to visualize
  """
  if data=="":
    pass
  else:
    g = graphviz.Source(data, format="png")
    g.render()
    image = Image.open("Source.gv.png", "r")
    plt.imshow(np.asarray(image))
    plt.show()