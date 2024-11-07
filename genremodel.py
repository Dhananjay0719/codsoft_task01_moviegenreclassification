import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
df = pd.read_csv('C:/Users/acer/Desktop/CODSOFT/Movie Genre classification/Genre Classification Dataset/train_data.txt', sep=' ::: ', engine='python', names=['Title', 'Genre', 'Description'], nrows=6000)

df.head()
df['Genre'].value_counts()
df = df[
    (df['Genre'] == 'drama') | (df['Genre'] == 'music') | (df['Genre'] == 'documentary') | (df['Genre'] == 'western')]
vec = TfidfVectorizer(stop_words='english')
matrix = vec.fit_transform(df['Description'])
X = matrix.toarray()
len(vec.get_feature_names_out())
vec.get_feature_names_out()
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], color=df['Genre'], hover_name=df['Title'], template='plotly_dark')
fig.update_layout(
    title="2 Component PCA visualization of Movie Genres",
    xaxis_title="1st Principal Component",
    yaxis_title="2nd Principal Component",
)
fig.show()
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)
fig = px.scatter_3d(x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2], color=df['Genre'], opacity=0.8,
                    title="3 Component PCA visualization of Movie Genres", hover_name=df['Title'], template='plotly_dark')
fig.show()