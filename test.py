import numpy as np
from collections import Counter
one = list([1,2,3,4,4,4,3])
num = np.array(one)

print(Counter(num.tolist()))
print(f'Num bins: {max(num.tolist())}')

