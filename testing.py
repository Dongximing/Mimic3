import pandas as pd
import numpy as np

# df = pd.DataFrame({"Name": ["Alice", "Bob", "Mallory", "Mallory", "Bob", "Mallory"],
#                  "City":["Seattle", "Seattle", "Portland", "Seattle", "Seattle", "Portland"],
#                  "Val":[4, 3, 3, np.nan, np.nan, 4]})
# print(df)
#
def index_default(i, char):
    """Returns the index of a character in a line, or the length of the string
    if the character does not appear.
    """
    try:
        retval = i.index(char)
    except ValueError:
        retval = 100000
    return retval

s = []

def split_log_line(i):
    """Splits a line at either a period or a colon, depending on which appears
    first in the line.
    """
    if index_default(i, "apple") < index_default(i, "banana"):
        a = i.split('apple')[1]
        # print(a)
        b = 'apple' + a
        s.append(b)

    elif index_default(i, "banana") < 100000:
        a = i.split('banana')[1]
        print(a)
        b = 'banana' + a
        s.append(b)
    else:
        s.append('')
text = ['apple are banana is','banana is apple are']
for i in text:
    split_log_line(i)
print(s)
