import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

a = stopwords.words('english')
print(a[:5])
