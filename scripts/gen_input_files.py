"""
Generate input files.
"""

import os
import json
import jinja2
import argparse

# Inputs
# =====================================
parser = argparse.ArgumentParser()
parser.add_argument("--inpfile", type=str, help="path to JSON input file")
args = parser.parse_args()

with open(args.inpfile) as file:
  glob_inp = json.load(file)

# Util functions
# =====================================
def render_json(template_file, glob_inp):
  with open(template_file, "r") as file:
    template = jinja2.Template(file.read())
  return json.loads(template.render(**glob_inp))

# Main
# =====================================
if (__name__ == "__main__"):

  print("Initialization ...")

  # Paths
  path_to_templates = "./input_files"
  path_to_saving = glob_inp["paths"]["saving"] + "/inputs/"
  os.makedirs(path_to_saving, exist_ok=True)

  # Render inputs
  for filename in os.listdir(path_to_templates):
    if filename.endswith(".json"):
      template_file = os.path.join(path_to_templates, filename)
      output_file = os.path.join(path_to_saving, filename)
      rendered = render_json(template_file, glob_inp)
      with open(output_file, "w") as file:
        json.dump(rendered, file, indent=2)

  print("Done!")
