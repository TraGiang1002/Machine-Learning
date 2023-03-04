import pandas as pd
df = pd.DataFrame(
    [(1, 2, 1, 6),
     (0, 3, 0, 7),
     (2, 0, 4, 3),
     (1, 1, 1, 4)], columns=['dogs', 'cats', 'bear', 'duck']
)
print(df.cov())