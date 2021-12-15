# ugly hack to add parent directory to jupyter
import sys
sys.path.append("../")


# Useful print output for jupyter
from IPython.display import clear_output
def cp(*args, **kwargs):
    clear_output(wait=True)
    print(*args, **kwargs)