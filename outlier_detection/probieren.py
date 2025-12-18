import outlier_detection.files as f
from outlier_detection.columns import NAME_DATA, NAME_OTHER

d = f.categorial_x_categorial()
vc = d.value_counts([NAME_DATA, NAME_OTHER])

# pdf = pd.DataFrame(columns=d[NAME_DATA].unique(), index=d[NAME_OTHER].unique())
# for row in vc.index:
#    pdf.loc[row[1], row[0]] = vc.loc[(row[0], row[1])]

pdf = vc.unstack()

print(pdf)

count = pdf.values.sum()


df = pdf.copy()
for row in vc.index:
    df.loc[row[1], row[0]] = (
        (pdf.loc[row[1]].sum() / count) * (pdf[row[0]].sum() / count) * count
    )

dff = pdf / df
print(pdf)

print(df)
print(dff)
