```python
import numpy as np
import pandas as pd
df = pd.read_csv('SMSSpamCollection.csv')
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ham\tGo until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham\tOk lar... Joking wif u oni...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>spam\tFree entry in 2 a wkly comp to win FA Cu...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ham\tU dun say so early hor... U c already the...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ham\tNah I don't think he goes to usf, he live...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>spam\tFreeMsg Hey there darling it's been 3 we...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5573 entries, 0 to 5572
    Data columns (total 1 columns):
     #   Column                                                                                                               Non-Null Count  Dtype 
    ---  ------                                                                                                               --------------  ----- 
     0   ham	Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...  5573 non-null   object
    dtypes: object(1)
    memory usage: 43.7+ KB
    


```python
df.sample(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ham\tGo until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4110</th>
      <td>ham\tYo, you gonna still be in stock tomorrow/...</td>
    </tr>
    <tr>
      <th>1194</th>
      <td>ham\tOk... C ya...</td>
    </tr>
    <tr>
      <th>5015</th>
      <td>ham\tI think the other two still need to get c...</td>
    </tr>
    <tr>
      <th>1112</th>
      <td>ham\tSo that means you still think of teju</td>
    </tr>
    <tr>
      <th>529</th>
      <td>ham\tJay says that you're a double-faggot</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (5573, 1)




```python
#1. Data Cleaning
#2. EDA - Exploratory Data Analysis
#3. Text Preprocessing
#4. Model Building
#5. Evaluation
#6. Improvemnts
#7. Website
#8. Deploy(heroku)
```

## 1. Data Cleaning


```python
import numpy as np
import pandas as pd

df = pd.read_csv('SMSSpamCollection.csv')
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5573 entries, 0 to 5572
    Data columns (total 1 columns):
     #   Column                                                                                                               Non-Null Count  Dtype 
    ---  ------                                                                                                               --------------  ----- 
     0   ham	Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...  5573 non-null   object
    dtypes: object(1)
    memory usage: 43.7+ KB
    


```python
df.sample(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ham\tGo until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3586</th>
      <td>spam\tI am hot n horny and willing I live loca...</td>
    </tr>
    <tr>
      <th>4922</th>
      <td>ham\tHi Dear Call me its urgnt. I don't know w...</td>
    </tr>
    <tr>
      <th>3166</th>
      <td>spam\tHOT LIVE FANTASIES call now 08707509020 ...</td>
    </tr>
    <tr>
      <th>4840</th>
      <td>spam\tPRIVATE! Your 2003 Account Statement for...</td>
    </tr>
    <tr>
      <th>1301</th>
      <td>ham\tI tot u reach liao. He said t-shirt.</td>
    </tr>
  </tbody>
</table>
</div>




```python
# drop last 3 cols
#df.drop(columns=['unnamed : 2','unnamed : 3','unnamed : 4' ],inplace = True)
df.sample(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ham\tGo until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4282</th>
      <td>ham\tU can call now...</td>
    </tr>
    <tr>
      <th>4372</th>
      <td>spam\tUr balance is now £600. Next question: C...</td>
    </tr>
    <tr>
      <th>3414</th>
      <td>ham\tNo pic. Please re-send.</td>
    </tr>
    <tr>
      <th>949</th>
      <td>ham\tIs that what time you want me to come?</td>
    </tr>
    <tr>
      <th>2745</th>
      <td>ham\tR ü going 4 today's meeting?</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Display the list of column names to verify their correctness
print(df.columns)

# Drop the columns
cols_to_drop = ['unnamed : 2', 'unnamed : 3', 'unnamed : 4']

# Check if the columns to drop exist in the DataFrame
missing_cols = [col for col in cols_to_drop if col not in df.columns]
if missing_cols:
    print("Columns not found:", missing_cols)
else:
    df.drop(columns=cols_to_drop, inplace=True)
    print("Columns dropped successfully.")

# Display a sample of the DataFrame after dropping columns
print(df.sample(5))

```

    Index(['ham\tGo until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...'], dtype='object')
    Columns not found: ['unnamed : 2', 'unnamed : 3', 'unnamed : 4']
         ham\tGo until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...
    185   ham\tHello handsome ! Are you finding that job...                                                                  
    2051  ham\tHey darlin.. i can pick u up at college i...                                                                  
    2005  ham\tCan't take any major roles in community o...                                                                  
    4708  ham\tDid you say bold, then torch later. Or on...                                                                  
    1663                   ham\tS but mostly not like that.                                                                  
    


```python
#renaming the columns
import pandas as pd

# Load the dataset
df = pd.read_csv('SMSSpamCollection.csv', sep='\t', header=None, names=['Label', 'Message'])

# Splitting the integers and labels
df['target'] = df['Label'].str.split('\t').str[0]
df['text'] = df['Label'].str.split('\t').str[1]
df = df[['target', 'text']]

#df = df.reset_index(drop=True)
# Display the resulting dataframe
print(df)

```

         target                                               text
    0       ham  Go until jurong point, crazy.. Available only ...
    1       ham                      Ok lar... Joking wif u oni...
    2      spam  Free entry in 2 a wkly comp to win FA Cup fina...
    3       ham  U dun say so early hor... U c already then say...
    4       ham  Nah I don't think he goes to usf, he lives aro...
    ...     ...                                                ...
    5569   spam  This is the 2nd time we have tried 2 contact u...
    5570    ham               Will ü b going to esplanade fr home?
    5571    ham  Pity, * was in mood for that. So...any other s...
    5572    ham  The guy did some bitching but I acted like i'd...
    5573    ham                         Rofl. Its true to its name
    
    [5574 rows x 2 columns]
    


```python
df.sample(25)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>target</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4622</th>
      <td>ham</td>
      <td>Received, understood n acted upon!</td>
    </tr>
    <tr>
      <th>4320</th>
      <td>ham</td>
      <td>Are you still playing with gautham?</td>
    </tr>
    <tr>
      <th>3756</th>
      <td>ham</td>
      <td>"Im on gloucesterroad what are uup to later?"</td>
    </tr>
    <tr>
      <th>3110</th>
      <td>ham</td>
      <td>Lol I was gonna last month. I cashed some in b...</td>
    </tr>
    <tr>
      <th>1109</th>
      <td>ham</td>
      <td>No you'll just get a headache trying to figure...</td>
    </tr>
    <tr>
      <th>3124</th>
      <td>ham</td>
      <td>He telling not to tell any one. If so treat fo...</td>
    </tr>
    <tr>
      <th>876</th>
      <td>spam</td>
      <td>Shop till u Drop, IS IT YOU, either 10K, 5K, £...</td>
    </tr>
    <tr>
      <th>4682</th>
      <td>ham</td>
      <td>Are you staying in town ?</td>
    </tr>
    <tr>
      <th>2166</th>
      <td>ham</td>
      <td>I'm not coming home 4 dinner.</td>
    </tr>
    <tr>
      <th>1054</th>
      <td>ham</td>
      <td>Jay's getting really impatient and belligerent</td>
    </tr>
    <tr>
      <th>575</th>
      <td>ham</td>
      <td>Nope i waiting in sch 4 daddy...</td>
    </tr>
    <tr>
      <th>713</th>
      <td>spam</td>
      <td>08714712388 between 10am-7pm Cost 10p</td>
    </tr>
    <tr>
      <th>886</th>
      <td>ham</td>
      <td>Gibbs unsold.mike hussey</td>
    </tr>
    <tr>
      <th>2870</th>
      <td>ham</td>
      <td>House-Maid is the murderer, coz the man was mu...</td>
    </tr>
    <tr>
      <th>3378</th>
      <td>ham</td>
      <td>Yup. Wun believe wat? U really neva c e msg i ...</td>
    </tr>
    <tr>
      <th>955</th>
      <td>spam</td>
      <td>Filthy stories and GIRLS waiting for your</td>
    </tr>
    <tr>
      <th>2384</th>
      <td>ham</td>
      <td>Your pussy is perfect!</td>
    </tr>
    <tr>
      <th>4064</th>
      <td>ham</td>
      <td>How are you. Its been ages. How's abj</td>
    </tr>
    <tr>
      <th>2492</th>
      <td>ham</td>
      <td>Greetings me, ! Consider yourself excused.</td>
    </tr>
    <tr>
      <th>2391</th>
      <td>ham</td>
      <td>First has she gained more than  &amp;lt;#&amp;gt; kg s...</td>
    </tr>
    <tr>
      <th>3061</th>
      <td>ham</td>
      <td>K..k...from tomorrow onwards started ah?</td>
    </tr>
    <tr>
      <th>2483</th>
      <td>ham</td>
      <td>Pansy! You've been living in a jungle for two ...</td>
    </tr>
    <tr>
      <th>4613</th>
      <td>ham</td>
      <td>Sorry da. I gone mad so many pending works wha...</td>
    </tr>
    <tr>
      <th>3909</th>
      <td>ham</td>
      <td>Sounds like a plan! Cardiff is still here and ...</td>
    </tr>
    <tr>
      <th>3601</th>
      <td>ham</td>
      <td>I know you mood off today</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
```


```python
 encoder.fit_transform(df['target'])
```




    array([0, 0, 1, ..., 0, 0, 0])




```python
df['target'] = encoder.fit_transform(df['target'])
#0 is assigned to ham
#1 is assigned to spam
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>target</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>Ok lar... Joking wif u oni...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>U dun say so early hor... U c already then say...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
    </tr>
  </tbody>
</table>
</div>




```python
#checking for missing values
df.isnull().sum()
```




    target    0
    text      0
    dtype: int64




```python
#check for duplicate values
df.duplicated().sum()
```




    414




```python
#remove duplicates
df = df.drop_duplicates(keep='first')
```


```python
df.duplicated().sum()
```




    0




```python
df.shape
```




    (5160, 2)



## 2. EDA : Exploratory Data Analysis


```python
df['target'].value_counts()
```




    0    4518
    1     642
    Name: target, dtype: int64




```python
import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(), labels=['ham','spam'], autopct="%0.2f")
```




    ([<matplotlib.patches.Wedge at 0x1847c675ad0>,
      <matplotlib.patches.Wedge at 0x1847c677e10>],
     [Text(-1.0170346463201791, 0.4190948916228736, 'ham'),
      Text(1.0170346267009303, -0.4190949392337011, 'spam')],
     [Text(-0.5547461707200977, 0.22859721361247648, '87.56'),
      Text(0.5547461600186891, -0.22859723958201877, '12.44')])




    
![png](output_24_1.png)
    



```python
#data is imbalanced

```


```python
import nltk
```


```python
!pip install nltk
```

    Requirement already satisfied: nltk in c:\users\happy\anaconda3\lib\site-packages (3.7)
    Requirement already satisfied: click in c:\users\happy\anaconda3\lib\site-packages (from nltk) (8.0.4)
    Requirement already satisfied: joblib in c:\users\happy\anaconda3\lib\site-packages (from nltk) (1.2.0)
    Requirement already satisfied: regex>=2021.8.3 in c:\users\happy\anaconda3\lib\site-packages (from nltk) (2022.7.9)
    Requirement already satisfied: tqdm in c:\users\happy\anaconda3\lib\site-packages (from nltk) (4.65.0)
    Requirement already satisfied: colorama in c:\users\happy\anaconda3\lib\site-packages (from click->nltk) (0.4.6)
    


```python
nltk.download('punkt')
```

    [nltk_data] Downloading package punkt to
    [nltk_data]     C:\Users\happy\AppData\Roaming\nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    




    True




```python
df['num_characters'] = df['text'].apply(len)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>target</th>
      <th>text</th>
      <th>num_characters</th>
      <th>num_words</th>
      <th>num_sentences</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
      <td>111</td>
      <td>24</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>Ok lar... Joking wif u oni...</td>
      <td>29</td>
      <td>8</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
      <td>155</td>
      <td>37</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>U dun say so early hor... U c already then say...</td>
      <td>49</td>
      <td>13</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
      <td>61</td>
      <td>15</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#num of words
df['num_words'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))

```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>target</th>
      <th>text</th>
      <th>num_characters</th>
      <th>num_words</th>
      <th>num_sentences</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
      <td>111</td>
      <td>24</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>Ok lar... Joking wif u oni...</td>
      <td>29</td>
      <td>8</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
      <td>155</td>
      <td>37</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>U dun say so early hor... U c already then say...</td>
      <td>49</td>
      <td>13</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
      <td>61</td>
      <td>15</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['num_sentences'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))
df.head() 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>target</th>
      <th>text</th>
      <th>num_characters</th>
      <th>num_words</th>
      <th>num_sentences</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
      <td>111</td>
      <td>24</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>Ok lar... Joking wif u oni...</td>
      <td>29</td>
      <td>8</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
      <td>155</td>
      <td>37</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>U dun say so early hor... U c already then say...</td>
      <td>49</td>
      <td>13</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
      <td>61</td>
      <td>15</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[['num_characters','num_words','num_sentences' ]].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>num_characters</th>
      <th>num_words</th>
      <th>num_sentences</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5160.000000</td>
      <td>5160.000000</td>
      <td>5160.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>79.141279</td>
      <td>18.578876</td>
      <td>1.951357</td>
    </tr>
    <tr>
      <th>std</th>
      <td>58.289387</td>
      <td>13.390839</td>
      <td>1.363466</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>36.000000</td>
      <td>9.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>61.000000</td>
      <td>15.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>118.000000</td>
      <td>26.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>910.000000</td>
      <td>220.000000</td>
      <td>28.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[df['target'] == 0]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>target</th>
      <th>text</th>
      <th>num_characters</th>
      <th>num_words</th>
      <th>num_sentences</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
      <td>111</td>
      <td>24</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>Ok lar... Joking wif u oni...</td>
      <td>29</td>
      <td>8</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>U dun say so early hor... U c already then say...</td>
      <td>49</td>
      <td>13</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
      <td>61</td>
      <td>15</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>Even my brother is not like to speak with me. ...</td>
      <td>77</td>
      <td>18</td>
      <td>2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5567</th>
      <td>0</td>
      <td>Huh y lei...</td>
      <td>12</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5570</th>
      <td>0</td>
      <td>Will ü b going to esplanade fr home?</td>
      <td>36</td>
      <td>9</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5571</th>
      <td>0</td>
      <td>Pity, * was in mood for that. So...any other s...</td>
      <td>57</td>
      <td>15</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5572</th>
      <td>0</td>
      <td>The guy did some bitching but I acted like i'd...</td>
      <td>125</td>
      <td>27</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5573</th>
      <td>0</td>
      <td>Rofl. Its true to its name</td>
      <td>26</td>
      <td>7</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>4518 rows × 5 columns</p>
</div>




```python
#ham
df[df['target'] == 0][['num_characters', 'num_words','num_sentences' ]].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>num_characters</th>
      <th>num_words</th>
      <th>num_sentences</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4518.000000</td>
      <td>4518.000000</td>
      <td>4518.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>70.860779</td>
      <td>17.279327</td>
      <td>1.806109</td>
    </tr>
    <tr>
      <th>std</th>
      <td>56.584730</td>
      <td>13.572536</td>
      <td>1.281858</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>34.000000</td>
      <td>8.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>53.000000</td>
      <td>13.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>91.000000</td>
      <td>22.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>910.000000</td>
      <td>220.000000</td>
      <td>28.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#spam
df[df['target'] == 1][['num_characters', 'num_words','num_sentences' ]].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>num_characters</th>
      <th>num_words</th>
      <th>num_sentences</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>642.000000</td>
      <td>642.000000</td>
      <td>642.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>137.414330</td>
      <td>27.724299</td>
      <td>2.973520</td>
    </tr>
    <tr>
      <th>std</th>
      <td>29.975596</td>
      <td>7.028380</td>
      <td>1.479211</td>
    </tr>
    <tr>
      <th>min</th>
      <td>13.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>131.000000</td>
      <td>25.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>148.000000</td>
      <td>29.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>157.000000</td>
      <td>32.000000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>223.000000</td>
      <td>46.000000</td>
      <td>8.000000</td>
    </tr>
  </tbody>
</table>
</div>



## Insights
### 1. The mean value of number of characters in spam is more the ham messages.
### 2. The mean value of number of words in spam is more the ham messages.
### 3. The mean value of number of sentences in spam is more the ham messages.


```python
#to see the comparison plot the histogram
import seaborn as sns

```


```python
plt.figure(figsize=(12,6))
sns.histplot(df[df['target']==0]['num_characters'])
sns.histplot(df[df['target']==1]['num_characters'],color='red')
print("Note : Red color for spam messages.")
```

    Note : Red color for spam messages.
    


    
![png](output_40_1.png)
    



```python
plt.figure(figsize=(12,6))
sns.histplot(df[df['target']==0]['num_words'])
sns.histplot(df[df['target']==1]['num_words'],color='red')
print("Note : Red color for spam messages.")
```

    Note : Red color for spam messages.
    


    
![png](output_41_1.png)
    



```python
sns.pairplot(df, hue = 'target')
```




    <seaborn.axisgrid.PairGrid at 0x1847e2649d0>




    
![png](output_42_1.png)
    



```python
#Corelation Coefficient
df.corr()
```

    C:\Users\happy\AppData\Local\Temp\ipykernel_5288\1939837678.py:2: FutureWarning: The default value of numeric_only in DataFrame.corr is deprecated. In a future version, it will default to False. Select only valid columns or specify the value of numeric_only to silence this warning.
      df.corr()
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>target</th>
      <th>num_characters</th>
      <th>num_words</th>
      <th>num_sentences</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>target</th>
      <td>1.000000</td>
      <td>0.376890</td>
      <td>0.257473</td>
      <td>0.282626</td>
    </tr>
    <tr>
      <th>num_characters</th>
      <td>0.376890</td>
      <td>1.000000</td>
      <td>0.966054</td>
      <td>0.637305</td>
    </tr>
    <tr>
      <th>num_words</th>
      <td>0.257473</td>
      <td>0.966054</td>
      <td>1.000000</td>
      <td>0.683835</td>
    </tr>
    <tr>
      <th>num_sentences</th>
      <td>0.282626</td>
      <td>0.637305</td>
      <td>0.683835</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.heatmap(df.corr(), annot=True)
```

    C:\Users\happy\AppData\Local\Temp\ipykernel_5288\621126171.py:1: FutureWarning: The default value of numeric_only in DataFrame.corr is deprecated. In a future version, it will default to False. Select only valid columns or specify the value of numeric_only to silence this warning.
      sns.heatmap(df.corr(), annot=True)
    




    <Axes: >




    
![png](output_44_2.png)
    


## 3. Data Preprocessing
* Lower case
* Tokenization
* Removing special characters
* Removing stop words and punctuation 
* Stemming


```python
#stop words --> that contribute in formation of sentence but doesn't provide meaning to sentence.
 #e.g : is, are etc.
```


```python
# 1. Lower Case

def transform_text(text):
    text = text.lower()
    return text
```


```python
transform_text('Hi how Are you')
```




    'hi how are you'




```python
# 2. Tokenization

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    return text
```


```python
transform_text('Hi how Are you')
```




    ['hi', 'how', 'are', 'you']




```python
# 3. Removing Special Characters

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text :
        if i.isalnum():
            y.append(i)
    return y
```


```python
transform_text('Hi how Are you 20% eg')
```




    ['hi', 'how', 'are', 'you', '20', 'eg']




```python
transform_text('Hi how Are you %% eg')
```




    ['hi', 'how', 'are', 'you', 'eg']




```python
import nltk
nltk.download('stopwords')
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     C:\Users\happy\AppData\Roaming\nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    




    True




```python
# 4. Removing stop words and punctuation 

from nltk.corpus import stopwords
stopwords.words('english')

 

```




    ['i',
     'me',
     'my',
     'myself',
     'we',
     'our',
     'ours',
     'ourselves',
     'you',
     "you're",
     "you've",
     "you'll",
     "you'd",
     'your',
     'yours',
     'yourself',
     'yourselves',
     'he',
     'him',
     'his',
     'himself',
     'she',
     "she's",
     'her',
     'hers',
     'herself',
     'it',
     "it's",
     'its',
     'itself',
     'they',
     'them',
     'their',
     'theirs',
     'themselves',
     'what',
     'which',
     'who',
     'whom',
     'this',
     'that',
     "that'll",
     'these',
     'those',
     'am',
     'is',
     'are',
     'was',
     'were',
     'be',
     'been',
     'being',
     'have',
     'has',
     'had',
     'having',
     'do',
     'does',
     'did',
     'doing',
     'a',
     'an',
     'the',
     'and',
     'but',
     'if',
     'or',
     'because',
     'as',
     'until',
     'while',
     'of',
     'at',
     'by',
     'for',
     'with',
     'about',
     'against',
     'between',
     'into',
     'through',
     'during',
     'before',
     'after',
     'above',
     'below',
     'to',
     'from',
     'up',
     'down',
     'in',
     'out',
     'on',
     'off',
     'over',
     'under',
     'again',
     'further',
     'then',
     'once',
     'here',
     'there',
     'when',
     'where',
     'why',
     'how',
     'all',
     'any',
     'both',
     'each',
     'few',
     'more',
     'most',
     'other',
     'some',
     'such',
     'no',
     'nor',
     'not',
     'only',
     'own',
     'same',
     'so',
     'than',
     'too',
     'very',
     's',
     't',
     'can',
     'will',
     'just',
     'don',
     "don't",
     'should',
     "should've",
     'now',
     'd',
     'll',
     'm',
     'o',
     're',
     've',
     'y',
     'ain',
     'aren',
     "aren't",
     'couldn',
     "couldn't",
     'didn',
     "didn't",
     'doesn',
     "doesn't",
     'hadn',
     "hadn't",
     'hasn',
     "hasn't",
     'haven',
     "haven't",
     'isn',
     "isn't",
     'ma',
     'mightn',
     "mightn't",
     'mustn',
     "mustn't",
     'needn',
     "needn't",
     'shan',
     "shan't",
     'shouldn',
     "shouldn't",
     'wasn',
     "wasn't",
     'weren',
     "weren't",
     'won',
     "won't",
     'wouldn',
     "wouldn't"]




```python
import string
string.punctuation
```




    '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'




```python
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text :
        if i.isalnum():
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text :
        if i not in stopwords.words('english') and i not in string.punctuation :
            y.append(i)
            
    
    return y
```


```python
transform_text('Hi how Are you Arpita?')
```




    ['hi', 'arpita']




```python
transform_text('Did you like my presentation on ML?')
```




    ['like', 'presentation', 'ml']




```python
df['text'][2000]
```




    "LMAO where's your fish memory when I need it?"




```python
# 5. Stemming

import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
nltk.download('punkt')
nltk.download('stopwords')
```

    [nltk_data] Downloading package punkt to
    [nltk_data]     C:\Users\happy\AppData\Roaming\nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    [nltk_data] Downloading package stopwords to
    [nltk_data]     C:\Users\happy\AppData\Roaming\nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    




    True




```python
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

nltk.download('punkt')
nltk.download('stopwords')

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text = y[:]
    y.clear()
    
    ps = PorterStemmer()  # Initialize the Porter Stemmer
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(ps.stem(i))
            
    return " ".join(y)
```

    [nltk_data] Downloading package punkt to
    [nltk_data]     C:\Users\happy\AppData\Roaming\nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    [nltk_data] Downloading package stopwords to
    [nltk_data]     C:\Users\happy\AppData\Roaming\nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    


```python
transform_text('I loved the YT lectures on Machine Learning. How about you?')
```




    'love yt lectur machin learn'




```python
df['text'][1991]
```




    'HI DARLIN IVE JUST GOT BACK AND I HAD A REALLY NICE NIGHT AND THANKS SO MUCH FOR THE LIFT SEE U TOMORROW XXX'




```python
transform_text('HI DARLIN IVE JUST GOT BACK AND I HAD A REALLY NICE NIGHT AND THANKS SO MUCH FOR THE LIFT SEE U TOMORROW XXX')
```




    'hi darlin ive got back realli nice night thank much lift see u tomorrow xxx'




```python
#creating a neww column for transform_text
df['transformed_text'] = df['text'].apply(transform_text)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>target</th>
      <th>text</th>
      <th>num_characters</th>
      <th>num_words</th>
      <th>num_sentences</th>
      <th>transformed_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
      <td>111</td>
      <td>24</td>
      <td>2</td>
      <td>go jurong point crazi avail bugi n great world...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>Ok lar... Joking wif u oni...</td>
      <td>29</td>
      <td>8</td>
      <td>2</td>
      <td>ok lar joke wif u oni</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
      <td>155</td>
      <td>37</td>
      <td>2</td>
      <td>free entri 2 wkli comp win fa cup final tkt 21...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>U dun say so early hor... U c already then say...</td>
      <td>49</td>
      <td>13</td>
      <td>1</td>
      <td>u dun say earli hor u c alreadi say</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
      <td>61</td>
      <td>15</td>
      <td>1</td>
      <td>nah think goe usf live around though</td>
    </tr>
  </tbody>
</table>
</div>




```python
pip install matplotlib

```

    Requirement already satisfied: matplotlib in c:\users\happy\anaconda3\lib\site-packages (3.7.1)
    Requirement already satisfied: contourpy>=1.0.1 in c:\users\happy\anaconda3\lib\site-packages (from matplotlib) (1.0.5)
    Requirement already satisfied: cycler>=0.10 in c:\users\happy\anaconda3\lib\site-packages (from matplotlib) (0.11.0)
    Requirement already satisfied: fonttools>=4.22.0 in c:\users\happy\anaconda3\lib\site-packages (from matplotlib) (4.25.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in c:\users\happy\anaconda3\lib\site-packages (from matplotlib) (1.4.4)
    Requirement already satisfied: numpy>=1.20 in c:\users\happy\anaconda3\lib\site-packages (from matplotlib) (1.24.3)
    Requirement already satisfied: packaging>=20.0 in c:\users\happy\anaconda3\lib\site-packages (from matplotlib) (23.0)
    Requirement already satisfied: pillow>=6.2.0 in c:\users\happy\anaconda3\lib\site-packages (from matplotlib) (9.4.0)
    Requirement already satisfied: pyparsing>=2.3.1 in c:\users\happy\anaconda3\lib\site-packages (from matplotlib) (3.0.9)
    Requirement already satisfied: python-dateutil>=2.7 in c:\users\happy\anaconda3\lib\site-packages (from matplotlib) (2.8.2)
    Requirement already satisfied: six>=1.5 in c:\users\happy\anaconda3\lib\site-packages (from python-dateutil>=2.7->matplotlib) (1.16.0)
    Note: you may need to restart the kernel to use updated packages.
    


```python
pip install wordcloud
```

    Requirement already satisfied: wordcloud in c:\users\happy\anaconda3\lib\site-packages (1.9.2)
    Requirement already satisfied: numpy>=1.6.1 in c:\users\happy\anaconda3\lib\site-packages (from wordcloud) (1.24.3)
    Requirement already satisfied: pillow in c:\users\happy\anaconda3\lib\site-packages (from wordcloud) (9.4.0)
    Requirement already satisfied: matplotlib in c:\users\happy\anaconda3\lib\site-packages (from wordcloud) (3.7.1)
    Requirement already satisfied: contourpy>=1.0.1 in c:\users\happy\anaconda3\lib\site-packages (from matplotlib->wordcloud) (1.0.5)
    Requirement already satisfied: cycler>=0.10 in c:\users\happy\anaconda3\lib\site-packages (from matplotlib->wordcloud) (0.11.0)
    Requirement already satisfied: fonttools>=4.22.0 in c:\users\happy\anaconda3\lib\site-packages (from matplotlib->wordcloud) (4.25.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in c:\users\happy\anaconda3\lib\site-packages (from matplotlib->wordcloud) (1.4.4)
    Requirement already satisfied: packaging>=20.0 in c:\users\happy\anaconda3\lib\site-packages (from matplotlib->wordcloud) (23.0)
    Requirement already satisfied: pyparsing>=2.3.1 in c:\users\happy\anaconda3\lib\site-packages (from matplotlib->wordcloud) (3.0.9)
    Requirement already satisfied: python-dateutil>=2.7 in c:\users\happy\anaconda3\lib\site-packages (from matplotlib->wordcloud) (2.8.2)
    Requirement already satisfied: six>=1.5 in c:\users\happy\anaconda3\lib\site-packages (from python-dateutil>=2.7->matplotlib->wordcloud) (1.16.0)
    Note: you may need to restart the kernel to use updated packages.
    


```python
from wordcloud import WordCloud
wc = WordCloud(width = 500, height = 500, min_font_size = 10, background_color = 'white' )
```


```python
spam_wc = wc.generate(df[df['target']==1]['transformed_text'].str.cat(sep = " "))
```


```python
plt.figure(figsize=(15,6))
plt.imshow(spam_wc)

#the message appeared in this is usually found in spam messages
```




    <matplotlib.image.AxesImage at 0x1847e354e50>




    
![png](output_72_1.png)
    



```python
ham_wc = wc.generate(df[df['target']==0]['transformed_text'].str.cat(sep = " "))
```


```python
plt.figure(figsize=(15,6))
plt.imshow(ham_wc)
```




    <matplotlib.image.AxesImage at 0x184018f4e50>




    
![png](output_74_1.png)
    



```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>target</th>
      <th>text</th>
      <th>num_characters</th>
      <th>num_words</th>
      <th>num_sentences</th>
      <th>transformed_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
      <td>111</td>
      <td>24</td>
      <td>2</td>
      <td>go jurong point crazi avail bugi n great world...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>Ok lar... Joking wif u oni...</td>
      <td>29</td>
      <td>8</td>
      <td>2</td>
      <td>ok lar joke wif u oni</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
      <td>155</td>
      <td>37</td>
      <td>2</td>
      <td>free entri 2 wkli comp win fa cup final tkt 21...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>U dun say so early hor... U c already then say...</td>
      <td>49</td>
      <td>13</td>
      <td>1</td>
      <td>u dun say earli hor u c alreadi say</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
      <td>61</td>
      <td>15</td>
      <td>1</td>
      <td>nah think goe usf live around though</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[df['target']==1]['transformed_text']
```




    2       free entri 2 wkli comp win fa cup final tkt 21...
    5       freemsg hey darl 3 week word back like fun sti...
    8       winner valu network custom select receivea pri...
    9       mobil 11 month u r entitl updat latest colour ...
    11      six chanc win cash 100 pound txt csh11 send co...
                                  ...                        
    5539    want explicit sex 30 sec ring 02073162414 cost...
    5542    ask 3mobil 0870 chatlin inclu free min india c...
    5549    contract mobil 11 mnth latest motorola nokia e...
    5568    remind o2 get pound free call credit detail gr...
    5569    2nd time tri 2 contact u pound prize 2 claim e...
    Name: transformed_text, Length: 642, dtype: object




```python
#converting the above result to list
df[df['target']==1]['transformed_text'].tolist()

#every message is a list of strings
```




    ['free entri 2 wkli comp win fa cup final tkt 21st may text fa 87121 receiv entri question std txt rate c appli 08452810075over18',
     'freemsg hey darl 3 week word back like fun still tb ok xxx std chg send rcv',
     'winner valu network custom select receivea prize reward claim call claim code kl341 valid 12 hour',
     'mobil 11 month u r entitl updat latest colour mobil camera free call mobil updat co free 08002986030',
     'six chanc win cash 100 pound txt csh11 send cost 6day tsandc appli repli hl 4 info',
     'urgent 1 week free membership prize jackpot txt word claim 81010 c lccltd pobox 4403ldnw1a7rw18',
     'xxxmobilemovieclub use credit click wap link next txt messag click http',
     'england v macedonia dont miss news txt ur nation team 87077 eg england 87077 tri wale scotland poboxox36504w45wq',
     'thank subscript rington uk mobil charg pleas confirm repli ye repli charg',
     '07732584351 rodger burn msg tri call repli sm free nokia mobil free camcord pleas call 08000930705 deliveri tomorrow',
     'sm ac sptv new jersey devil detroit red wing play ice hockey correct incorrect end repli end sptv',
     'congrat 1 year special cinema pass 2 call 09061209465 c suprman v matrix3 starwars3 etc 4 free 150pm dont miss',
     'valu custom pleas advis follow recent review mob award bonu prize call 09066364589',
     'urgent ur award complimentari trip eurodisinc trav aco entry41 claim txt di 87121 morefrmmob shracomorsglsuplt 10 ls1 3aj',
     'hear new divorc barbi come ken stuff',
     'pleas call custom servic repres 0800 169 6031 guarante cash prize',
     'free rington wait collect simpli text password mix 85069 verifi get usher britney fml po box 5249 mk17 92h 450ppw 16',
     'gent tri contact last weekend draw show prize guarante call claim code k52 valid 12hr 150ppm',
     'winner u special select 2 receiv 4 holiday flight inc speak live oper 2 claim',
     'privat 2004 account statement 07742676969 show 786 unredeem bonu point claim call 08719180248 identifi code 45239 expir',
     'urgent mobil award bonu caller prize final tri contact u call landlin 09064019788 box42wr29c 150ppm',
     'today voda number end 7548 select receiv 350 award match pleas call 08712300220 quot claim code 4041 standard rate app',
     'sunshin quiz wkli q win top soni dvd player u know countri algarv txt ansr 82277 sp tyron',
     'want 2 get laid tonight want real dog locat sent direct 2 ur mob join uk largest dog network bt txting gravel 69888 nt ec2a 150p',
     'rcv msg chat svc free hardcor servic text go 69988 u get noth u must age verifi yr network tri',
     'freemsg repli text randi sexi femal live local luv hear netcollex ltd 08700621170150p per msg repli stop end',
     'custom servic annonc new year deliveri wait pleas call 07046744435 arrang deliveri',
     'winner u special select 2 receiv cash 4 holiday flight inc speak live oper 2 claim 0871277810810',
     'stop bootydeli invit friend repli see stop send stop frnd 62468',
     'bangbab ur order way u receiv servic msg 2 download ur content u goto wap bangb tv ur mobil menu',
     'urgent tri contact last weekend draw show prize guarante call claim code s89 valid 12hr',
     'pleas call custom servic repres freephon 0808 145 4742 guarante cash prize',
     'uniqu enough find 30th august',
     '500 new mobil 2004 must go txt nokia 89545 collect today 2optout',
     'u meet ur dream partner soon ur career 2 flyng start 2 find free txt horo follow ur star sign horo ari',
     'text meet someon sexi today u find date even flirt join 4 10p repli name age eg sam 25 18 recd thirtyeight penc',
     'u 447801259231 secret admir look 2 make contact r reveal think ur 09058094597',
     'congratul ur award 500 cd voucher 125gift guarante free entri 2 100 wkli draw txt music 87066 tnc',
     'tri contact repli offer video handset 750 anytim network min unlimit text camcord repli call 08000930705',
     'hey realli horni want chat see nake text hot 69698 text charg 150pm unsubscrib text stop 69698',
     'ur rington servic chang 25 free credit go choos content stop txt club stop 87070 club4 po box1146 mk45 2wt',
     'rington club get uk singl chart mobil week choos top qualiti rington messag free charg',
     'hmv bonu special 500 pound genuin hmv voucher answer 4 easi question play send hmv 86688 info',
     'custom may claim free camera phone upgrad pay go sim card loyalti call 0845 021 end c appli',
     'sm ac blind date 4u rodds1 aberdeen unit kingdom check http sm blind date send hide',
     'themob check newest select content game tone gossip babe sport keep mobil fit funki text wap 82468',
     'think ur smart win week weekli quiz text play 85222 cs winnersclub po box 84 m26 3uz',
     'decemb mobil entitl updat latest colour camera mobil free call mobil updat co free 08002986906',
     'call germani 1 penc per minut call fix line via access number 0844 861 85 prepay direct access',
     'valentin day special win quiz take partner trip lifetim send go 83600 rcvd',
     'fanci shag txt xxuk suzi txt cost per msg tnc websit x',
     'ur current 500 pound maxim ur send cash 86688 cc 08708800282',
     'xma offer latest motorola sonyericsson nokia free bluetooth doubl min 1000 txt orang call mobileupd8 08000839402',
     'discount code rp176781 stop messag repli stop custom servic 08717205546',
     'thank rington order refer t91 charg gbp 4 per week unsubscrib anytim call custom servic 09057039994',
     'doubl min txt 4 6month free bluetooth orang avail soni nokia motorola phone call mobileupd8 08000839402',
     '4mth half price orang line rental latest camera phone 4 free phone 11mth call mobilesdirect free 08000938767 updat or2stoptxt',
     'free rington text first 87131 poli text get 87131 true tone help 0845 2814032 16 1st free tone txt stop',
     '100 date servic cal l 09064012103 box334sk38ch',
     'free entri weekli competit text word win 80086 18 c',
     'send logo 2 ur lover 2 name join heart txt love name1 name2 mobno eg love adam eve 07123456789 87077 yahoo pobox36504w45wq txtno 4 ad 150p',
     'someon contact date servic enter phone fanci find call landlin 09111032124 pobox12n146tf150p',
     'urgent mobil number award prize guarante call 09058094455 land line claim valid 12hr',
     'congrat nokia 3650 video camera phone call 09066382422 call cost 150ppm ave call 3min vari mobil close 300603 post bcm4284 ldn wc1n3xx',
     'loan purpos homeown tenant welcom previous refus still help call free 0800 1956669 text back',
     'upgrdcentr orang custom may claim free camera phone upgrad loyalti call 0207 153 offer end 26th juli c appli avail',
     'okmail dear dave final notic collect 4 tenerif holiday 5000 cash award call 09061743806 landlin tc sae box326 cw25wx 150ppm',
     'want 2 get laid tonight want real dog locat sent direct 2 ur mob join uk largest dog network txting moan 69888nyt ec2a 150p',
     'free messag activ 500 free text messag repli messag word free term condit visit',
     'congratul week competit draw u prize claim call 09050002311 b4280703 sm 18 150ppm',
     'guarante latest nokia phone 40gb ipod mp3 player prize txt word collect 83355 ibhltd ldnw15h',
     'boltblu tone 150p repli poli mono eg poly3 cha cha slide yeah slow jamz toxic come stop 4 tone txt',
     'credit top http renew pin tgxxrz',
     'urgent mobil award bonu caller prize 2nd attempt contact call box95qu',
     'today offer claim ur worth discount voucher text ye 85023 savamob member offer mobil cs 08717898035 sub 16 unsub repli x',
     'reciev tone within next 24hr term condit pleas see channel u teletext pg 750',
     'privat 2003 account statement 07815296484 show 800 point call 08718738001 identifi code 41782 expir',
     'monthlysubscript csc web age16 2stop txt stop',
     'cash prize claim call09050000327',
     'mobil number claim call us back ring claim hot line 09050005321',
     'tri contact repli offer 750 min 150 textand new video phone call 08002988890 repli free deliveri tomorrow',
     'ur chanc win wkli shop spree txt shop c custcar 08715705022',
     'special select receiv 2000 pound award call 08712402050 line close cost 10ppm cs appli ag promo',
     'privat 2003 account statement 07753741225 show 800 point call 08715203677 identifi code 42478 expir',
     'import custom servic announc call freephon 0800 542 0825',
     'xclusiv clubsaisai 2morow soire special zouk nichol rose 2 ladi info',
     '22 day kick euro2004 u kept date latest news result daili remov send get txt stop 83222',
     'new textbuddi chat 2 horni guy ur area 4 25p free 2 receiv search postcod txt one name 89693',
     'today vodafon number end 4882 select receiv award number match call 09064019014 receiv award',
     'dear voucher holder 2 claim week offer pc go http ts cs stop text txt stop 80062',
     'privat 2003 account statement show 800 point call 08715203694 identifi code 40533 expir',
     'cash prize claim call09050000327 c rstm sw7 3ss 150ppm',
     '88800 89034 premium phone servic call 08718711108',
     'sm ac sun0819 post hello seem cool want say hi hi stop send stop 62468',
     'get ur 1st rington free repli msg tone gr8 top 20 tone phone everi week per wk 2 opt send stop 08452810071 16',
     'hi sue 20 year old work lapdanc love sex text live bedroom text sue textoper g2 1da 150ppmsg',
     'forward 448712404000 pleas call 08712404000 immedi urgent messag wait',
     'review keep fantast nokia game deck club nokia go 2 unsubscrib alert repli word',
     '4mth half price orang line rental latest camera phone 4 free phone call mobilesdirect free 08000938767 updat or2stoptxt cs',
     '08714712388 cost 10p',
     'urgent 2nd attempt contact u u call 09071512433 b4 050703 csbcm4235wc1n3xx callcost 150ppm mobilesvari 50',
     'guarante cash prize claim yr prize call custom servic repres 08714712394',
     'email alertfrom jeri stewarts 2kbsubject prescripiton drvgsto listen email call 123',
     'hi custom loyalti offer new nokia6650 mobil txtauction txt word start 81151 get 4t ctxt tc',
     'u subscrib best mobil content servic uk per 10 day send stop helplin 08706091795',
     'realiz 40 year thousand old ladi run around tattoo',
     'import custom servic announc premier',
     'romant pari 2 night 2 flight book 4 next year call 08704439680t cs appli',
     'urgent ur guarante award still unclaim call 09066368327 claimcod m39m51',
     'ur award citi break could win summer shop spree everi wk txt store 88039 skilgm tscs087147403231winawk age16',
     'import custom servic announc premier call freephon 0800 542 0578',
     'ever thought live good life perfect partner txt back name age join mobil commun',
     '5 free top polyphon tone call 087018728737 nation rate get toppoli tune sent everi week text subpoli 81618 per pole unsub 08718727870',
     'orang custom may claim free camera phone upgrad loyalti call 0207 153 offer end 14thmarch c appli availa',
     'last chanc claim ur worth discount voucher today text shop 85023 savamob offer mobil cs savamob pobox84 m263uz sub 16',
     'free 1st week no1 nokia tone 4 ur mobil everi week txt nokia 8077 get txting tell ur mate pobox 36504 w45wq',
     'guarante award even cashto claim ur award call free 08000407165 2 stop getstop 88222 php rg21 4jx',
     'congratul ur award either cd gift voucher free entri 2 weekli draw txt music 87066 tnc',
     'u outbid simonwatson5120 shinco dvd plyr 2 bid visit sm 2 end bid notif repli end',
     'smsservic yourinclus text credit pl goto 3qxj9 unsubscrib stop extra charg help 9ae',
     '25p 4 alfi moon children need song ur mob tell ur m8 txt tone chariti 8007 nokia poli chariti poli zed 08701417012 profit 2 chariti',
     'u secret admir reveal think u r special call opt repli reveal stop per msg recd cust care 07821230901',
     'dear voucher holder claim week offer pc pleas go http ts cs appli stop text txt stop 80062',
     'want 750 anytim network min 150 text new video phone five pound per week call 08002888812 repli deliveri tomorrow',
     'tri contact offer new video phone 750 anytim network min half price rental camcord call 08000930705 repli deliveri wed',
     'last chanc 2 claim ur worth discount ye 85023 offer mobil cs 08717898035 sub 16 remov txt x stop',
     'urgent call 09066350750 landlin complimentari 4 ibiza holiday cash await collect sae cs po box 434 sk3 8wp 150 ppm',
     'talk sexi make new friend fall love world discreet text date servic text vip 83110 see could meet',
     'congratul ur award either yr suppli cd virgin record mysteri gift guarante call 09061104283 ts cs approx 3min',
     'privat 2003 account statement 07808 xxxxxx show 800 point call 08719899217 identifi code 41685 expir',
     'hello need posh bird chap user trial prod champney put need address dob asap ta r',
     'u want xma 100 free text messag new video phone half price line rental call free 0800 0721072 find',
     'shop till u drop either 10k 5k cash travel voucher call ntt po box cr01327bt fixedlin cost 150ppm mobil vari',
     'sunshin quiz wkli q win top soni dvd player u know countri liverpool play mid week txt ansr 82277 sp tyron',
     'u secret admir look 2 make contact r reveal think ur 09058094565',
     'u secret admir look 2 make contact r reveal think ur',
     'remind download content alreadi paid goto http mymobi collect content',
     'lastest stereophon marley dizze racal libertin stroke win nookii game flirt click themob wap bookmark text wap 82468',
     'januari male sale hot gay chat cheaper call nation rate cheap peak stop text call 08712460324',
     'money r lucki winner 2 claim prize text money 2 88600 give away text rate box403 w1t1ji',
     'dear matthew pleas call 09063440451 landlin complimentari 4 lux tenerif holiday cash await collect ppm150 sae cs box334 sk38xh',
     'urgent call 09061749602 landlin complimentari 4 tenerif holiday cash await collect sae cs box 528 hp20 1yf 150ppm',
     'get touch folk wait compani txt back name age opt enjoy commun',
     'ur current 500 pound maxim ur send go 86688 cc 08718720201 po box',
     'filthi stori girl wait',
     'urgent tri contact today draw show prize guarante call 09050001808 land line claim m95 valid12hr',
     'congrat 2 mobil 3g videophon r call 09063458130 videochat wid mate play java game dload polyph music nolin rentl',
     'panason bluetoothhdset free nokia free motorola free doublemin doubletxt orang contract call mobileupd8 08000839402 call 2optout',
     'free 1st week no1 nokia tone 4 ur mob everi week txt nokia 8007 get txting tell ur mate pobox 36504 w45wq',
     'guess somebodi know secretli fanci wan na find give us call 09065394514 landlin datebox1282essexcm61xn 18',
     'know someon know fanci call 09058097218 find pobox 6 ls15hb 150p',
     '1000 flirt txt girl bloke ur name age eg girl zoe 18 8007 join get chat',
     '18 day euro2004 kickoff u kept inform latest news result daili unsubscrib send get euro stop 83222',
     'eastend tv quiz flower dot compar violet tulip lili txt e f 84025 4 chanc 2 win cash',
     'new local date area lot new peopl regist area repli date start 18 replys150',
     'someon u know ask date servic 2 contact cant guess call 09058091854 reveal po box385 m6 6wu',
     'urgent tri contact today draw show prize guarante call 09050003091 land line claim c52 valid12hr',
     'dear u invit xchat final attempt contact u txt chat 86688',
     'award sipix digit camera call 09061221061 landlin deliveri within 28day cs box177 m221bp 2yr warranti 150ppm 16 p',
     'win urgent mobil number award prize guarante call 09061790121 land line claim 3030 valid 12hr 150ppm',
     'dear subscrib ur draw 4 gift voucher b enter receipt correct an elvi presley birthday txt answer 80062',
     'messag import inform o2 user today lucki day 2 find log onto http fantast surpris await',
     '449050000301 price claim call 09050000301',
     'bore speed date tri speedchat txt speedchat 80155 like em txt swap get new chatter chat80155 pobox36504w45wq rcd 16',
     'want 750 anytim network min 150 text new video phone five pound per week call 08000776320 repli deliveri tomorrow',
     'take part mobil survey yesterday 500 text 2 use howev wish 2 get txt send txt 80160 c',
     'ur hmv quiz current maxim ur send hmv1 86688',
     'dont forget place mani free request wish inform call 08707808226',
     'know u u know send chat 86688 let find rcvd ldn 18 year',
     'thank winner notifi sm good luck futur market repli stop 84122 custom servic 08450542832',
     '1000 girl mani local 2 u r virgin 2 r readi 2 4fil ur everi sexual need u 4fil text cute 69911',
     'got take 2 take part wrc ralli oz u lucozad energi text ralli le 61200 25p see pack itcould u',
     'sex ur mobil free sexi pic jordan text babe everi wk get sexi celeb 4 pic 16 087016248',
     '1 new voicemail pleas call 08719181503',
     'win year suppli cd 4 store ur choic worth enter weekli draw txt music 87066 ts cs',
     'sim subscrib select receiv bonu get deliv door txt word ok 88600 claim exp 30apr',
     '1 new voicemail pleas call 08719181513',
     '1 nokia tone 4 ur mob everi week txt nok 87021 1st tone free get txtin tell ur friend 16 repli hl 4info',
     'repli name address receiv post week complet free accommod variou global locat',
     'free entri weekli comp send word enter 84128 18 c cust care 08712405020',
     'pleas call 08712402779 immedi urgent messag wait',
     'hungri gay guy feel hungri 4 call 08718730555 stop text call 08712460324',
     'u get 2 phone wan na chat 2 set meet call 09096102316 u cum 2moro luv jane xx',
     'network oper servic free c visit',
     'enjoy jamster videosound gold club credit 2 new get fun help call 09701213186',
     'get 3 lion england tone repli lionm 4 mono lionp 4 poli 4 go 2 origin n best tone 3gbp network oper rate appli',
     'win newest harri potter order phoenix book 5 repli harri answer 5 question chanc first among reader',
     'ur balanc ur next question sang girl 80 2 answer txt ur answer good luck',
     'free2day sexi st georg day pic jordan txt pic 89080 dont miss everi wk sauci celeb 4 pic c 0870241182716',
     'hot live fantasi call 08707509020 20p per min ntt ltd po box 1327 croydon cr9 5wb 0870 k',
     'bear pic nick tom pete dick fact type tri gay chat photo upload call 08718730666 2 stop text call 08712460324',
     '500 new mobil 2004 must go txt nokia 89545 collect today 2optout txtauction',
     'doubl min doubl txt price linerent latest orang bluetooth mobil call mobileupd8 latest offer 08000839402',
     'urgent import inform o2 user today lucki day 2 find log onto http fantast surpris await',
     'dear u invit xchat final attempt contact u txt chat 86688 ldn 18 yr',
     'congratul ur award either cd gift voucher free entri 2 weekli draw txt music 87066 tnc 1 win150ppmx3age16',
     'sale arsen dartboard good condit doubl trebl',
     'free 1st week entri 2 textpod 4 chanc 2 win 40gb ipod cash everi wk txt pod 84128 ts cs custcar 08712405020',
     'regist optin subscrib ur draw 4 gift voucher enter receipt correct an 80062 what no1 bbc chart',
     'summer final fanci chat flirt sexi singl yr area get match repli summer free 2 join optout txt stop help08714742804',
     'clair havin borin time alon u wan na cum 2nite chat 09099725823 hope 2 c u luv clair xx',
     'bought one rington get text cost 3 pound offer tone etc',
     '09066362231 urgent mobil 07xxxxxxxxx bonu caller prize 2nd attempt reach call 09066362231 asap',
     '07801543489 guarante latest nokia phone 40gb ipod mp3 player prize txt word collect',
     'hi luci hubbi meetin day fri b alon hotel u fanci cumin pl leav msg 2day 09099726395 luci x',
     'account credit 500 free text messag activ txt word credit 80488 cs',
     'sm ac jsco energi high u may know 2channel 2day ur leadership skill r strong psychic repli an end repli end jsco',
     'hot live fantasi call 08707509020 20p per min ntt ltd po box 1327 croydon cr9 5wb 0870 nation rate call',
     'thank vote sing along star karaok mobil free link repli sing',
     'brand new mobil music servic live free music player arriv shortli instal phone brows content top artist',
     'urgent mobil award bonu caller prize 2nd attempt contact call box95qu bt nation rate',
     'nokia 7250i get win free auction take part send nokia 86021',
     'hello orang 1 month free access game news sport plu 10 free text 20 photo messag repli ye term appli',
     'ur current 500 pound maxim ur send go 86688 cc 08718720201',
     'sm auction brand new nokia 7250 4 auction today auction free 2 join take part txt nokia 86021',
     'privat 2003 account statement show 800 point call 08719899230 identifi code 41685 expir',
     'regist subscrib yr draw 4 gift voucher b enter receipt correct an next olymp txt an 80062',
     'urgent mobil number award prize guarante call 09061790121 land line claim valid 12hr 150ppm',
     'pro video club need help info call 08701237397 must club credit redeem enjoy',
     'u secret admir look 2 make contact r reveal think ur 09058094599',
     '500 free text msg text ok 80488 credit account',
     'select stay 1 250 top british hotel noth holiday worth claim call london bx 526 sw73ss',
     'eeri nokia tone 4u rpli tone titl 8007 eg tone dracula 8007 titl ghost addamsfa munster exorcist twilight pobox36504w45wq 150p',
     '0a network allow compani bill sm respons supplier shop give guarante sell g',
     'freemsg feelin kinda lnli hope u like 2 keep compani jst got cam mobi wan na c pic txt repli date 82242 msg150p 2rcv hlp 08712317606 stop 82242',
     'ur chanc win cash everi wk txt action c custcar 08712405022',
     'rgent 2nd attempt contact u u call 09071512433 b4 050703 csbcm4235wc1n3xx callcost 150ppm mobilesvari 50',
     'hi ur lookin 4 sauci daytim fun wiv busti marri woman free next week chat 2 sort time 09099726429 janinexx',
     'urgent tri contact today draw show prize guarante call 09050001295 land line claim a21 valid 12hr',
     'monthli password wap use wap phone pc',
     'today vodafon number end 0089 last four digit select receiv award number match pleas call 09063442151 claim award',
     'free top rington weekli 1st week subpoli 3 per',
     'free msg sorri servic order 81303 could deliv suffici credit pleas top receiv servic',
     'hard live 121 chat choos girl connect live call 09094646899 cheap chat uk biggest live servic vu bcm1896wc1n3xx',
     'wow boy r back take 2007 uk tour win vip ticket vip club txt club trackmarqu ltd info vipclub4u',
     'hi mandi sullivan call hotmix fm chosen receiv easter prize draw pleas telephon 09041940223 claim prize transfer someon els',
     'ur go 2 bahama callfreefon 08081560665 speak live oper claim either bahama cruis cash opt txt x 07786200117',
     'someon conact date servic enter phone fanci find call landlin pobox12n146tf15',
     'hi 07734396839 ibh custom loyalti offer new nokia6600 mobil txtauction txt word start get 4t',
     'sm auction nokia 7250i get win free auction take part send nokia 86021',
     'call freephon 0800 542 0578',
     'buy space invad 4 chanc 2 win orig arcad game consol press 0 game arcad std wap charg see 4 term set purchas',
     'big brother alert comput select u 10k cash 150 voucher call ntt po box cro1327 bt landlin cost 150ppm mobil vari',
     'win winner foley ipod excit prize soon keep eye ur mobil visit',
     'today voda number end 1225 select receiv match pleas call 08712300220 quot claim code 3100 standard rate app',
     'hottest pic straight phone see get wet want xx text pic 89555 txt cost 150p textoper g696ga 18 xxx',
     'hack chat get backdoor entri 121 chat room fraction cost repli neo69 call 09050280520 subscrib 25p pm dp bcm box 8027 ldn wc1n3xx',
     'free nokia motorola upto 12mth linerent 500 free min free call mobileupd8 08001950382 call',
     '2nd time tri 2 contact u 750 pound prize 2 claim easi call 08718726970 10p per min',
     'guarante cash claim yr prize call custom servic repres',
     'would like see xxx pic hot nearli ban uk',
     'u secret admir look 2 make contact r reveal think ur 09058094594',
     'dear 0776xxxxxxx u invit xchat final attempt contact u txt chat 86688 ldn 18yr',
     'urgent pleas call 09061743811 landlin abta complimentari 4 tenerif holiday cash await collect sae cs box 326 cw25wx 150ppm',
     'call 09090900040 listen extrem dirti live chat go offic right total privaci one know sic listen 60p min',
     'freemsg hey u got 1 fone repli wild txt ill send u pic hurri im bore work xxx 18 stop2stop',
     'free entri 2 weekli comp chanc win ipod txt pod 80182 get entri std txt rate c appli 08452810073 detail',
     'new textbuddi chat 2 horni guy ur area 4 25p free 2 receiv search postcod txt one name 89693 08715500022 rpl stop 2 cnl',
     'call 08702490080 tell u 2 call 09066358152 claim prize u 2 enter ur mobil person detail prompt care',
     'free 1st week entri 2 textpod 4 chanc 2 win 40gb ipod cash everi wk txt vpod 81303 ts cs custcar 08712405020',
     'peopl dog area call 09090204448 join like mind guy arrang 1 1 even minapn ls278bb',
     'well done 4 costa del sol holiday await collect call 09050090044 toclaim sae tc pobox334 stockport sk38xh max10min',
     'guess somebodi know secretli fanci wan na find give us call 09065394973 landlin datebox1282essexcm61xn 18',
     '500 free text messag valid 31 decemb 2005',
     'guarante award even cashto claim ur award call free 08000407165 2 stop getstop 88222 php',
     'repli win weekli 2006 fifa world cup held send stop 87239 end servic',
     'urgent pleas call 09061743810 landlin abta complimentari 4 tenerif holiday 5000 cash await collect sae cs box 326 cw25wx 150 ppm',
     'free tone hope enjoy new content text stop 61610 unsubscrib provid',
     'themob yo yo come new select hot download member get free click open next link sent ur fone',
     'great news call freefon 08006344447 claim guarante cash gift speak live oper',
     'u win music gift voucher everi week start txt word draw 87066 tsc',
     'call 09094100151 use ur min call cast mob vari servic provid aom aom box61 m60 1er u stop age',
     'urgent mobil bonu caller prize 2nd attempt reach call 09066362220 asap box97n7qp 150ppm',
     'sexi singl wait text age follow gender wither f gay men text age follow',
     'freemsg claim ur 250 sm ok 84025 use web2mobil 2 ur mate etc join c box139 la32wu 16 remov txtx stop',
     '85233 free rington repli real',
     'well done england get offici poli rington colour flag yer mobil text tone flag 84199 txt eng stop box39822 w111wx',
     'final chanc claim ur worth discount voucher today text ye 85023 savamob member offer mobil cs savamob pobox84 m263uz sub 16',
     'sm servic inclus text credit pl goto unsubscrib stop extra charg po box420 ip4 5we',
     'winner special select receiv cash award speak live oper claim call cost 10p',
     'sunshin hol claim ur med holiday send stamp self address envelop drink us uk po box 113 bray wicklow eir quiz start saturday unsub stop',
     'u win music gift voucher everi week start txt word draw 87066 tsc skillgam 1winaweek age16 150ppermesssubscript',
     'b4u voucher marsm log onto discount credit opt repli stop custom care call 08717168528',
     'freemsg hey buffi 25 love satisfi men home alon feel randi repli 2 c pix qlynnbv help08700621170150p msg send stop stop txt',
     'free 1st week no1 nokia tone 4 ur mob everi week txt nokia 87077 get txting tell ur mate zed pobox 36504 w45wq',
     'free camera phone linerent 750 cross ntwk min price txt bundl deal also avbl call 08001950382 mf',
     'urgent mobil 07xxxxxxxxx bonu caller prize 2nd attempt reach call 09066362231 asap box97n7qp 150ppm',
     'urgent 4 costa del sol holiday await collect call 09050090044 toclaim sae tc pobox334 stockport sk38xh max10min',
     'guarante cash prize claim yr prize call custom servic repres 08714712379 cost 10p',
     'thank rington order ref number k718 mobil charg tone arriv pleas call custom servic 09065069120',
     'hi ya babe x u 4goten bout scammer get smart though regular vodafon respond get prem rate no use also bewar',
     'back 2 work 2morro half term u c 2nite 4 sexi passion b4 2 go back chat 09099726481 luv dena call',
     'thank rington order ref number r836 mobil charg tone arriv pleas call custom servic 09065069154',
     'splashmobil choos 1000 gr8 tone wk subscrit servic weekli tone cost 300p u one credit kick back enjoy',
     'heard u4 call 4 rude chat privat line 01223585334 cum wan 2c pic gettin shag text pix 8552 2end send stop 8552 sam xxx',
     'forward 88877 free entri weekli comp send word enter 88877 18 c',
     '88066 88066 lost 3pound help',
     'mobil 11mth updat free orang latest colour camera mobil unlimit weekend call call mobil upd8 freefon 08000839402 2stoptx',
     '1 new messag pleas call 08718738034',
     'forward 21870000 hi mailbox messag sm alert 4 messag 21 match pleas call back 09056242159 retriev messag match',
     'mobi pub high street prize u know new duchess cornwal txt first name stop 008704050406 sp arrow',
     'congratul thank good friend u xma prize 2 claim easi call 08718726971 10p per minut',
     'tddnewslett game thedailydraw dear helen dozen free game great prizeswith',
     'urgent mobil number bonu caller prize 2nd attempt reach call 09066368753 asap box 97n7qp 150ppm',
     'doubl min txt orang price linerent motorola sonyericsson free call mobileupd8 08000839402',
     'download mani rington u like restrict 1000 2 choos u even send 2 yr buddi txt sir 80082',
     'pleas call 08712402902 immedi urgent messag wait',
     'spook mob halloween collect logo pic messag plu free eeri tone txt card spook 8007 zed 08701417012150p per',
     'fantasi footbal back tv go sky gamestar sky activ play dream team score start saturday regist sky opt 88088',
     'tone club sub expir 2 repli monoc 4 mono polyc 4 poli 1 weekli 150p per week txt stop 2 stop msg free stream 0871212025016',
     'xma prize draw tri contact today draw show prize guarante call 09058094565 land line valid 12hr',
     'ye place town meet excit adult singl uk txt chat 86688',
     'someon contact date servic enter phone becausethey fanci find call landlin pobox1 w14rg 150p',
     'babe u want dont u babi im nasti thing 4 filthyguy fanci rude time sexi bitch go slo n hard txt xxx slo 4msg',
     'sm servic inclus text credit pl gotto login 3qxj9 unsubscrib stop extra charg help 08702840625 9ae',
     'valentin day special win quiz take partner trip lifetim send go 83600 rcvd',
     'guess first time creat web page read wrote wait opinion want friend',
     'ur chanc win cash everi wk txt play c custcar 08715705022',
     'sppok ur mob halloween collect nokia logo pic messag plu free eeri tone txt card spook 8007',
     'urgent call 09066612661 landlin complementari 4 tenerif holiday cash await collect sae cs po box 3 wa14 2px 150ppm sender hol offer',
     'winner valu network custom hvae select receiv reward collect call valid 24 hour acl03530150pm',
     'u nokia 6230 plu free digit camera u get u win free auction take part send nokia 83383 16',
     'free entri weekli comp send word win 80086 18 c',
     'text82228 get rington logo game question info',
     'freemsg award free mini digit camera repli snap collect prize quizclub opt stop sp rwm',
     'messag brought gmw connect',
     'congrat 2 mobil 3g videophon r call 09063458130 videochat wid ur mate play java game dload polyph music nolin rentl bx420 ip4 5we 150p',
     'next amaz xxx picsfree1 video sent enjoy one vid enough 2day text back keyword picsfree1 get next video',
     'u subscrib best mobil content servic uk per ten day send stop helplin 08706091795',
     '3 free tarot text find love life tri 3 free text chanc 85555 16 3 free msg',
     'join uk horniest dog servic u sex 2nite sign follow instruct txt entri 69888 150p',
     'knock knock txt whose 80082 enter r weekli draw 4 gift voucher 4 store yr choic cs age16',
     'forward 21870000 hi mailbox messag sm alert 40 match pleas call back 09056242159 retriev messag match',
     'free ring tone text poli everi week get new tone 0870737910216yr',
     'urgent mobil 077xxx bonu caller prize 2nd attempt reach call 09066362206 asap box97n7qp 150ppm',
     'guarante latest nokia phone 40gb ipod mp3 player prize txt word collect 83355 ibhltd ldnw15h',
     'hello darl today would love chat dont tell look like sexi',
     '8007 free 1st week no1 nokia tone 4 ur mob everi week txt nokia 8007 get txting tell ur mate pobox 36504 w4 5wq norm',
     'wan na get laid 2nite want real dog locat sent direct ur mobil join uk largest dog network txt park 69696 nyt ec2a 3lp',
     'tri contact respons offer new nokia fone camcord hit repli call 08000930705 deliveri',
     'new tone week includ 1 ab 2 sara 3 order follow instruct next messag',
     'urgent tri contact today draw show prize guarante call 09050003091 land line claim c52 valid 12hr',
     'sport fan get latest sport news str 2 ur mobil 1 wk free plu free tone txt sport 8007 norm',
     'urgent urgent 800 free flight europ give away call b4 10th sept take friend 4 free call claim ba128nnfwfly150ppm',
     '88066 lost help',
     'freemsg fanci flirt repli date join uk fastest grow mobil date servic msg rcvd 25p optout txt stop repli date',
     'great new offer doubl min doubl txt best orang tariff get latest camera phone 4 free call mobileupd8 free 08000839402 2stoptxt cs',
     'hope enjoy new content text stop 61610 unsubscrib provid',
     'urgent pleas call 09066612661 landlin cash luxuri 4 canari island holiday await collect cs sae award 20m12aq 150ppm',
     'urgent pleas call 09066612661 landlin complimentari 4 lux costa del sol holiday cash await collect ppm 150 sae cs jame 28 eh74rr',
     'marri local women look discreet action 5 real match instantli phone text match 69969 msg cost 150p 2 stop txt stop bcmsfwc1n3xx',
     'burger king wan na play footi top stadium get 2 burger king 1st sept go larg super walk winner',
     'come take littl time child afraid dark becom teenag want stay night',
     'ur chanc win cash everi wk txt action c custcar 08712405022',
     'u bin award play 4 instant cash call 08715203028 claim everi 9th player win min optout 08718727870',
     'freemsg fav xma tone repli real',
     'gr8 poli tone 4 mob direct 2u rpli poli titl 8007 eg poli breathe1 titl crazyin sleepingwith finest ymca pobox365o4w45wq 300p',
     'interflora late order interflora flower christma call 0800 505060 place order midnight tomorrow',
     'romcapspam everyon around respond well presenc sinc warm outgo bring real breath sunshin',
     'congratul thank good friend u xma prize 2 claim easi call 08712103738 10p per minut',
     'send logo 2 ur lover 2 name join heart txt love name1 name2 mobno eg love adam eve 07123456789 87077 yahoo pobox36504w45wq txtno 4 ad 150p',
     'tkt euro2004 cup final cash collect call 09058099801 b4190604 pobox 7876150ppm',
     'jamster get crazi frog sound poli text mad1 real text mad2 88888 6 crazi sound 3 c appli',
     'chanc realiti fantasi show call 08707509020 20p per min ntt ltd po box 1327 croydon cr9 5wb 0870 nation rate call',
     'adult 18 content video shortli',
     'chanc realiti fantasi show call 08707509020 20p per min ntt ltd po box 1327 croydon cr9 5wb 0870 nation rate call',
     'hey boy want hot xxx pic sent direct 2 ur phone txt porn 69855 24hr free 50p per day stop text stopbcm sf wc1n3xx',
     'doubl min 1000 txt orang tariff latest motorola sonyericsson nokia bluetooth free call mobileupd8 08000839402 yhl',
     'ur current 500 pound maxim ur send cash 86688 cc 08718720201 po box',
     'urgent mobil number award prize guarante call 09058094454 land line claim valid 12hr',
     'sorri u unsubscrib yet mob offer packag min term 54 week pl resubmit request expiri repli themob help 4 info',
     '1 new messag pleas call 08712400200',
     'current messag await collect collect messag call 08718723815',
     'urgent mobil award bonu caller prize final attempt 2 contact u call 08714714011',
     'ever notic drive anyon go slower idiot everyon drive faster maniac',
     'xma offer latest motorola sonyericsson nokia free bluetooth dvd doubl min 1000 txt orang call mobileupd8 08000839402',
     'repli win weekli profession sport tiger wood play send stop 87239 end servic',
     '1 polyphon tone 4 ur mob everi week txt pt2 87575 1st tone free get txtin tell ur friend 16 repli hl 4info',
     'messag free welcom new improv sex dog club unsubscrib servic repli stop msg 150p',
     '12mth half price orang line rental 400min call mobileupd8 08000839402',
     'free unlimit hardcor porn direct 2 mobil txt porn 69200 get free access 24 hr chrgd 50p per day txt stop 2exit msg free',
     'unsubscrib servic get ton sexi babe hunk straight phone go http subscript',
     'hi babe jordan r u im home abroad lone text back u wan na chat xxsp text stop stopcost 150p 08712400603',
     'get brand new mobil phone agent mob plu load goodi info text mat 87021',
     'lord ring return king store repli lotr 2 june 4 chanc 2 win lotr soundtrack cd stdtxtrate repli stop end txt',
     'good luck draw take place 28th feb good luck remov send stop 87239 custom servic 08708034412',
     '1st wk free gr8 tone str8 2 u wk txt nokia 8007 classic nokia tone hit 8007 poli',
     'lookatm thank purchas video clip lookatm charg 35p think better send video mmsto 32323',
     'sexi sexi cum text im wet warm readi porn u fun msg free recd msg 150p inc vat 2 cancel text stop',
     '2nd time tri contact u prize claim call 09053750005 b4 sm 08718725756 140ppm',
     'dear voucher holder claim week offer pc pleas go http ts cs appli',
     '2nd time tri 2 contact u 750 pound prize 2 claim easi call 08712101358 10p per min',
     'ur award citi break could win summer shop spree everi wk txt store',
     'urgent tri contact today draw show prize guarante call 09066358361 land line claim y87 valid 12hr',
     'thank rington order refer number x29 mobil charg tone arriv pleas call custom servic 09065989180',
     'ur current 500 pound maxim ur send collect 83600 cc 08718720201 po box',
     'congratul thank good friend u xma prize 2 claim easi call 08718726978 10p per minut',
     '44 7732584351 want new nokia 3510i colour phone deliveredtomorrow 300 free minut mobil 100 free text free camcord repli call 08000930705',
     'someon u know ask date servic 2 contact cant guess call 09058097189 reveal pobox 6 ls15hb 150p',
     'camera award sipix digit camera call 09061221066 fromm landlin deliveri within 28 day',
     'today voda number end 5226 select receiv 350 award hava match pleas call 08712300220 quot claim code 1131 standard rate app',
     'messag free welcom new improv sex dog club unsubscrib servic repli stop msg 150p 18',
     'rct thnq adrian u text rgd vatian',
     'contact date servic someon know find call land line pobox45w2tg150p',
     'sorri miss call let talk time 07090201529',
     'complimentari 4 star ibiza holiday cash need urgent collect 09066364349 landlin lose',
     'free msg bill mobil number mistak shortcod call 08081263000 charg call free bt landlin',
     'pleas call 08712402972 immedi urgent messag wait',
     'urgent mobil number award bonu caller prize call 09058095201 land line valid 12hr',
     'want new nokia 3510i colour phone deliveredtomorrow 300 free minut mobil 100 free text free camcord repli call 08000930705',
     'life never much fun great came made truli special wo forget enjoy one',
     'want new video phone 600 anytim network min 400 inclus video call download 5 per week free deltomorrow call 08002888812 repli',
     'valu custom pleas advis follow recent review mob award bonu prize call 09066368470',
     'welcom pleas repli age gender begin 24m',
     'freemsg unlimit free call activ smartcal txt call unlimit call help 08448714184 stop txt stop landlineonli',
     'mobil 10 mth updat latest orang phone free save free call text ye callback orno opt',
     'new 2 club dont fink met yet b gr8 2 c u pleas leav msg 2day wiv ur area 09099726553 repli promis carli x lkpobox177hp51fl',
     'camera award sipix digit camera call 09061221066 fromm landlin deliveri within 28 day',
     'get free mobil video player free movi collect text go free extra film order c appli 18 yr',
     'save money wed lingeri choos superb select nation deliveri brought weddingfriend',
     'heard u4 call night knicker make beg like u last time 01223585236 xx luv',
     'bloomberg center wait appli futur http',
     'want new video phone750 anytim network min 150 text five pound per week call 08000776320 repli deliveri tomorrow',
     'contact date servic someon know find call land line pobox45w2tg150p',
     'wan2 win westlif 4 u m8 current tour 1 unbreak 2 untam 3 unkempt text 3 cost 50p text',
     'dorothi bank granit issu explos pick member 300 nasdaq symbol cdgt per',
     'winner guarante caller prize final attempt contact claim call 09071517866 150ppmpobox10183bhamb64x',
     'xma new year eve ticket sale club day 10am till 8pm thur fri sat night week sell fast',
     'rock yr chik get 100 filthi film xxx pic yr phone rpli filth saristar ltd e14 9yt 08701752560 450p per 5 day stop2 cancel',
     'next month get upto 50 call 4 ur standard network charg 2 activ call 9061100010 c 1st4term pobox84 m26 3uz cost min mobcudb',
     'urgent tri contact u today draw show prize guarante call 09050000460 land line claim j89 po box245c2150pm',
     'text banneduk 89555 see cost 150p textoper g696ga xxx',
     'auction round highest bid next maximum bid bid send bid 10 bid good luck',
     'collect valentin weekend pari inc flight hotel prize guarante text pari',
     'custom loyalti offer new nokia6650 mobil txtauction txt word start 81151 get 4t ctxt tc',
     'wo believ true incred txt repli g learn truli amaz thing blow mind o2fwd',
     'hot n horni will live local text repli hear strt back 150p per msg netcollex ltdhelpdesk 02085076972 repli stop end',
     'want new nokia 3510i colour phone deliv tomorrow 200 free minut mobil 100 free text free camcord repli call 08000930705',
     'congratul winner august prize draw call 09066660100 prize code 2309',
     '8007 25p 4 alfi moon children need song ur mob tell ur m8 txt tone chariti 8007 nokia poli chariti poli zed 08701417012 profit 2 chariti',
     'get offici england poli rington colour flag yer mobil tonight game text tone flag optout txt eng stop box39822 w111wx',
     'custom servic announc recent tri make deliveri unabl pleas call 07090298926',
     'stop club tone repli stop mix see html term club tone cost mfl po box 1146 mk45 2wt',
     'wamma get laid want real doggin locat sent direct mobil join uk largest dog network txt dog 69696 nyt ec2a 3lp',
     'promot number 8714714 ur award citi break could win summer shop spree everi wk txt store 88039 skilgm tscs087147403231winawk age16',
     'winner special select receiv cash award speak live oper claim call cost 10p',
     'thank rington order refer number x49 mobil charg tone arriv pleas call custom servic text txtstar',
     'hi 2night ur lucki night uve invit 2 xchat uk wildest chat txt chat 86688 ldn 18yr',
     '146tf150p',
     'dear voucher holder 2 claim 1st class airport loung pass use holiday voucher call book quot 1st class x 2',
     'someon u know ask date servic 2 contact cant guess call 09058095107 reveal pobox 7 s3xi 150p',
     'mila age23 blond new uk look sex uk guy u like fun text mtalk 1st 5free increment help08718728876',
     'claim 200 shop spree call 08717895698 mobstorequiz10ppm',
     'want funk ur fone weekli new tone repli tones2u 2 text origin n best tone 3gbp network oper rate appli',
     'twink bear scalli skin jock call miss weekend fun call 08712466669 2 stop text call 08712460324 nat rate',
     'tri contact repli offer video handset 750 anytim network min unlimit text camcord repli call 08000930705',
     'urgent tri contact last weekend draw show prize guarante call claim code k61 valid 12hour',
     '74355 xma iscom ur award either cd gift voucher free entri 2 r weekli draw txt music 87066 tnc',
     'congratul u claim 2 vip row ticket 2 c blu concert novemb blu gift guarante call 09061104276 claim ts cs',
     'free msg singl find partner area 1000 real peopl wait chat send chat 62220cncl send stopc per msg',
     'win newest potter order phoenix book 5 repli harri answer 5 question chanc first among reader',
     'free msg rington http wml 37819',
     'oh god found number glad text back xafter msg cst std ntwk chg',
     'link pictur sent also use http',
     'doubl min 1000 txt orang tariff latest motorola sonyericsson nokia bluetooth free call mobileupd8 08000839402',
     'urgent 2nd attempt contact prize yesterday still await collect claim call acl03530150pm',
     'dear dave final notic collect 4 tenerif holiday 5000 cash award call 09061743806 landlin tc sae box326 cw25wx 150ppm',
     'tell u 2 call 09066358152 claim prize u 2 enter ur mobil person detail prompt care',
     '2004 account 07xxxxxxxxx show 786 unredeem point claim call 08719181259 identifi code xxxxx expir',
     'want new video handset 750 anytim network min half price line rental camcord repli call 08000930705 deliveri tomorrow',
     'free rington repli real poli eg real1 pushbutton dontcha babygoodby golddigg webeburnin 1st tone free 6 u join',
     'free msg get gnarl barkley crazi rington total free repli go messag right',
     'refus loan secur unsecur ca get credit call free 0800 195 6669 text back',
     'special select receiv 3000 award call 08712402050 line close cost 10ppm cs appli ag promo',
     'valu vodafon custom comput pick win prize collect easi call 09061743386',
     'free video camera phone half price line rental 12 mth 500 cross ntwk min 100 txt call mobileupd8 08001950382',
     'ringtonek 84484',
     'rington club gr8 new poli direct mobil everi week',
     'bank granit issu explos pick member 300 nasdaq symbol cdgt per',
     'bore housew chat n date rate landlin',
     'tri call repli sm video mobil 750 min unlimit text free camcord repli call 08000930705 del thur',
     '2nd time tri contact u prize 2 claim easi call 087104711148 10p per minut',
     'receiv week tripl echo rington shortli enjoy',
     'u select stay 1 250 top british hotel noth holiday valu dial 08712300220 claim nation rate call bx526 sw73ss',
     'chosen receiv award pl call claim number 09066364311 collect award select receiv valu mobil custom',
     'win cash prize prize worth',
     'thank rington order refer number mobil charg tone arriv pleas call custom servic 09065989182',
     'mobi pub high street prize u know new duchess cornwal txt first name stop 008704050406 sp',
     'week savamob member offer access call 08709501522 detail savamob pobox 139 la3 2wu savamob offer mobil',
     'contact date servic someon know find call mobil landlin 09064017305 pobox75ldns7',
     'chase us sinc sept definit pay thank inform ignor kath manchest',
     'loan purpos even bad credit tenant welcom call 08717111821',
     '87077 kick new season 2wk free goal news ur mobil txt ur club name 87077 eg villa 87077',
     'orang bring rington time chart hero free hit week go rington pic wap stop receiv tip repli stop',
     'privat 2003 account statement 07973788240 show 800 point call 08715203649 identifi code 40533 expir',
     'tri call repli sm video mobil 750 min unlimit text free camcord repli call 08000930705',
     'gsoh good spam ladi u could b male gigolo 2 join uk fastest grow men club repli oncal mjzgroup repli stop msg',
     'hot live fantasi call 08707500020 20p per min ntt ltd po box 1327 croydon cr9 5wb 0870 nation rate call',
     'urgent mobil number award ukp 2000 prize guarante call 09061790125 landlin claim valid 12hr 150ppm',
     'spjanuari male sale hot gay chat cheaper call nation rate cheap peak stop text call 08712460324',
     'freemsg today day readi horni live town love sex fun game netcollex ltd 08700621170150p per msg repli stop end',
     'simpson movi releas juli 2007 name band die start film day day day send b c',
     'pleas call amanda regard renew upgrad current handset free charg offer end today tel 0845 021 3680 subject c',
     'want new video phone 750 anytim network min half price line rental free text 3 month repli call 08000930705 free deliveri',
     'dear voucher holder claim week offer pc pleas go http ts cs appli',
     'urgent pleas call abta complimentari 4 spanish holiday cash await collect sae cs box 47 po19 2ez 150ppm',
     'cmon babe make horni turn txt fantasi babe im hot sticki need repli cost 2 cancel send stop',
     'import inform 4 orang user 0796xxxxxx today ur lucki day 2 find log onto http fantast prizeawait',
     'miss call alert number call left messag 07008009200',
     'freemsg record indic may entitl 3750 pound accid claim free repli ye msg opt text stop',
     'show ur colour euro 2004 offer get england flag 3lion tone ur phone click follow servic messag info',
     'text pass 69669 collect polyphon rington normal gpr charg appli enjoy tone',
     'accordingli repeat text word ok mobil phone send',
     'block breaker come delux format new featur great graphic buy repli get bbdelux take challeng',
     'import inform 4 orang user today lucki day 2find log onto http fantast surpris await',
     'natalja invit friend repli see stop send stop frnd 62468',
     'urgent import inform 02 user today lucki day 2 find log onto http fantast surpris await',
     'kit strip bill 150p netcollex po box 1013 ig11 oja',
     'pleas call 08712402578 immedi urgent messag wait',
     'let send free anonym mask messag im send messag see potenti abus',
     'congrat 2 mobil 3g videophon r call 09061744553 videochat wid ur mate play java game dload polyh music nolin rentl bx420 ip4 5we 150pm',
     'import inform 4 orang user 0789xxxxxxx today lucki day 2find log onto http fantast surpris await',
     'date servic ask 2 contact u someon shi call 09058091870 reveal pobox84 m26 3uz 150p',
     'want new video handset 750 time network min unlimit text camcord repli call 08000930705 del sat',
     'ur balanc next question complet landmark big bob barri ben text b c good luck',
     'ur tonex subscript renew charg choos 10 poli month bill msg',
     'prize go anoth custom c polo ltd suit 373 london w1j 6hl pleas call back busi',
     'want new nokia 3510i colour phone deliv tomorrow 200 free minut mobil 100 free text free camcord repli call 8000930705',
     'recpt order rington order process',
     'one regist subscrib u enter draw 4 100 gift voucher repli enter unsubscrib text stop',
     'chanc win free bluetooth headset simpli repli back adp',
     'b floppi b snappi happi gay chat servic photo upload call 08718730666 2 stop text call 08712460324',
     'welcom msg free give free call futur mg bill 150p daili cancel send go stop 89123',
     'receiv mobil content enjoy',
     'want explicit sex 30 sec ring 02073162414 cost',
     'latest nokia mobil ipod mp3 player proze guarante repli win 83355 norcorp',
     'sm servic inclus text credit pl goto 3qxj9 unsubscrib stop extra charg help 9ae',
     'mobil club choos top qualiti item mobil 7cfca1a',
     'money wine number 946 wot next',
     'want cock hubbi away need real man 2 satisfi txt wife 89938 string action txt stop 2 end txt rec otbox 731 la1 7w',
     'gr8 new servic live sex video chat mob see sexiest dirtiest girl live ur phone 4 detail text horni 89070 cancel send stop 89070',
     'freemsg hi babi wow got new cam mobi wan na c hot pic fanci chat im w8in 4utxt rpli chat 82242 hlp 08712317606 msg150p 2rcv',
     'wan na laugh tri mobil logon txting word chat send 8883 cm po box 4217 london w1a 6zf rcvd',
     'urgent 2nd attempt contact u u 09071512432 b4 300603t 50',
     'congratul ur award 500 cd voucher 125gift guarante free entri 2 100 wkli draw txt music 87066',
     'contract mobil 11 mnth latest motorola nokia etc free doubl min text orang tariff text ye callback remov record',
     'u secret admir look 2 make contact r reveal think ur',
     'freemsg txt call 86888 claim reward 3 hour talk time use phone inc 3hr 16 stop txtstop',
     'sunshin quiz win super soni dvd record cannam capit australia text mquiz b',
     'today voda number end 7634 select receiv reward match pleas call 08712300220 quot claim code 7684 standard rate appli',
     'rip get mobil content call 08717509990 six download 3',
     'tri contact repli offer video phone 750 anytim network min half price line rental camcord repli call 08000930705',
     'xma reward wait comput randomli pick loyal mobil custom receiv reward call 09066380611',
     'privat 2003 account statement show 800 point call 08718738002 identifi code 48922 expir',
     'custom servic announc recent tri make deliveri unabl pleas call 07099833605',
     'hi babe chloe r u smash saturday night great weekend u miss sp text stop stop',
     'urgent mobil 07808726822 award bonu caller prize 2nd attempt contact call box95qu',
     'free game get rayman golf 4 free o2 game arcad 1st get ur game set repli post save activ8 press 0 key arcad termsappli',
     'mobil 10 mth updat latest phone free keep ur number get extra free text ye call',
     'weekli tone readi download week new tone includ 1 crazi f 2 3 black p info n',
     'get lot cash weekend dear welcom weekend got biggest best ever cash give away',
     'thank 4 continu support question week enter u in2 draw 4 cash name new us presid txt an 80082',
     'uniqu user id remov send stop 87239 custom servic 08708034412',
     'urgent 09066649731from landlin complimentari 4 ibiza holiday cash await collect sae cs po box 434 sk3 8wp 150ppm',
     'urgent 2nd attempt contact prize yesterday still await collect claim call 09061702893',
     'santa call would littl one like call santa xma eve call 09077818151 book time last 3min 30 c',
     'privat 2004 account statement 078498 7 show 786 unredeem bonu point claim call 08719180219 identifi code 45239 expir',
     'check choos babe video fgkslpopw fgkslpo',
     'u r winner u ave special select 2 receiv cash 4 holiday flight inc speak live oper 2 claim 18',
     'new mobil 2004 must go txt nokia 89545 collect today 2optout txtauction',
     'privat 2003 account statement show 800 point call 08715203652 identifi code 42810 expir',
     'free messag thank use auction subscript servic 18 2 skip auction txt 2 unsubscrib txt stop customercar 08718726270',
     'lyricalladi invit friend repli see stop send stop frnd 62468',
     'want latest video handset 750 anytim network min half price line rental repli call 08000930705 deliveri tomorrow',
     'ou guarante latest nokia phone 40gb ipod mp3 player prize txt word collect 83355 ibhltd ldnw15h',
     'free polyphon rington text super 87131 get free poli tone week 16 sn pobox202 nr31 7z subscript 450pw',
     'warner villag 83118 c colin farrel swat wkend warner villag get 1 free med popcorn show c c kiosk repli soni 4 mre film offer',
     'goal arsen 4 henri 7 v liverpool 2 henri score simpl shot 6 yard pass bergkamp give arsen 2 goal margin 78 min',
     'hi sexychat girl wait text text great night chat send stop stop servic',
     'hi ami send free phone number coupl day give access adult parti',
     'welcom select o2 servic ad benefit call special train advisor free mobil diall 402',
     'dear voucher holder next meal us use follow link pc 2 enjoy 2 4 1 dine experiencehttp',
     'urgent tri contact today draw show prize guarante call 09058094507 land line claim valid 12hr',
     'donat unicef asian tsunami disast support fund text donat 864233 ad next bill',
     'goldvik invit friend repli see stop send stop frnd 62468',
     'phoni award today voda number end xxxx select receiv award match pleas call 08712300220 quot claim code 3100 standard rate app',
     'cd 4u congratul ur award cd gift voucher gift guarante freeentri 2 wkli draw xt music 87066 tnc',
     'guarante cash prize claim yr prize call custom servic repres 08714712412 cost 10p',
     'ur current 500 pound maxim ur send go 86688 cc 08718720201',
     'privat 2003 account statement show 800 point call 08715203685 identifi expir',
     'like tell deepest darkest fantasi call 09094646631 stop text call 08712460324 nat rate',
     'natali invit friend repli see stop send stop frnd 62468',
     'jamster get free wallpap text heart 88888 c appli 16 need help call 08701213186',
     'free video camera phone half price line rental 12 mth 500 cross ntwk min 100 txt call mobileupd8 08001950382',
     '83039 uk break accommodationvouch term condit appli 2 claim mustprovid claim number 15541',
     '5p 4 alfi moon children need song ur mob tell ur m8 txt tone chariti 8007 nokia poli chariti poli zed 08701417012 profit 2 chariti',
     'win shop spree everi week start 2 play text store skilgm tscs08714740323 1winawk age16',
     '2nd attempt contract u week top prize either cash prize call 09066361921',
     'want new nokia 3510i colour phone deliveredtomorrow 300 free minut mobil 100 free text free camcord repli call 08000930705',
     'themob hit link get premium pink panther game new 1 sugabab crazi zebra anim badass hoodi 4 free',
     'msg mobil content order resent previou attempt fail due network error queri customersqueri',
     '1 new messag pleas call 08715205273',
     'decemb mobil entitl updat latest colour camera mobil free call mobil updat vco free 08002986906',
     'get 3 lion england tone repli lionm 4 mono lionp 4 poli 4 go 2 origin n best tone 3gbp network oper rate appli',
     'privat 2003 account statement 078',
     '4 costa del sol holiday await collect call 09050090044 toclaim sae tc pobox334 stockport sk38xh max10min',
     'get garden readi summer free select summer bulb seed worth scotsman saturday stop go2',
     'sm auction brand new nokia 7250 4 auction today auction free 2 join take part txt nokia 86021',
     'ree entri 2 weekli comp chanc win ipod txt pod 80182 get entri std txt rate c appli 08452810073 detail',
     'record indic u mayb entitl 5000 pound compens accid claim 4 free repli claim msg 2 stop txt stop',
     'call germani 1 penc per minut call fix line via access number 0844 861 85 prepay direct access',
     'mobil 11mth updat free orang latest colour camera mobil unlimit weekend call call mobil upd8 freefon 08000839402 2stoptxt',
     'privat 2003 account statement fone show 800 point call 08715203656 identifi code 42049 expir',
     'someonon know tri contact via date servic find could call mobil landlin 09064015307 box334sk38ch',
     'urgent pleas call 09061213237 landlin cash 4 holiday await collect cs sae po box 177 m227xi',
     'urgent mobil number award prize guarante call 09061790126 land line claim valid 12hr 150ppm',
     'urgent pleas call 09061213237 landlin cash luxuri 4 canari island holiday await collect cs sae po box m227xi 150ppm',
     'xma iscom ur award either cd gift voucher free entri 2 r weekli draw txt music 87066 tnc',
     'u r subscrib 2 textcomp 250 wkli comp 1st wk free question follow subsequ wk charg unsubscrib txt stop 2 84128 custcar 08712405020',
     'call 09095350301 send girl erot ecstaci stop text call 08712460324 nat rate',
     'import messag final contact attempt import messag wait custom claim dept expir call 08717507382',
     'date two start sent text talk sport radio last week connect think coincid',
     'current lead bid paus auction send custom care 08718726270',
     'free entri gr8prize wkli comp 4 chanc win latest nokia 8800 psp cash everi great 80878 08715705022',
     '1 new messag call',
     'santa call would littl one like call santa xma eve call 09058094583 book time',
     'guarante 32000 award mayb even cash claim ur award call free 0800 legitimat efreefon number wat u think',
     'latest news polic station toilet stolen cop noth go',
     'sparkl shop break 45 per person call 0121 2025050 visit',
     'txt call 86888 claim reward 3 hour talk time use phone inc 3hr 16 stop txtstop',
     'wml c',
     'urgent last weekend draw show cash spanish holiday call 09050000332 claim c rstm sw7 3ss 150ppm',
     'urgent tri contact last weekend draw show u prize guarante call 09064017295 claim code k52 valid 12hr 150p pm',
     '2p per min call germani 08448350055 bt line 2p per min check info c text stop opt',
     'marvel mobil play offici ultim game ur mobil right text spider 83338 game send u free 8ball wallpap',
     'privat 2003 account statement 07808247860 show 800 point call 08719899229 identifi code 40411 expir',
     'privat 2003 account statement show 800 point call 08718738001 identifi code 49557 expir',
     'want explicit sex 30 sec ring 02073162414 cost gsex pobox 2667 wc1n 3xx',
     'ask 3mobil 0870 chatlin inclu free min india cust serv sed ye l8er got mega bill 3 dont giv shit bailiff due day 3 want',
     'contract mobil 11 mnth latest motorola nokia etc free doubl min text orang tariff text ye callback remov record',
     'remind o2 get pound free call credit detail great offer pl repli 2 text valid name hous postcod',
     '2nd time tri 2 contact u pound prize 2 claim easi call 087187272008 now1 10p per minut']




```python
#printing all the messages
for msg in df[df['target']==1]['transformed_text'].tolist():
    
    print(msg)
```

    free entri 2 wkli comp win fa cup final tkt 21st may text fa 87121 receiv entri question std txt rate c appli 08452810075over18
    freemsg hey darl 3 week word back like fun still tb ok xxx std chg send rcv
    winner valu network custom select receivea prize reward claim call claim code kl341 valid 12 hour
    mobil 11 month u r entitl updat latest colour mobil camera free call mobil updat co free 08002986030
    six chanc win cash 100 pound txt csh11 send cost 6day tsandc appli repli hl 4 info
    urgent 1 week free membership prize jackpot txt word claim 81010 c lccltd pobox 4403ldnw1a7rw18
    xxxmobilemovieclub use credit click wap link next txt messag click http
    england v macedonia dont miss news txt ur nation team 87077 eg england 87077 tri wale scotland poboxox36504w45wq
    thank subscript rington uk mobil charg pleas confirm repli ye repli charg
    07732584351 rodger burn msg tri call repli sm free nokia mobil free camcord pleas call 08000930705 deliveri tomorrow
    sm ac sptv new jersey devil detroit red wing play ice hockey correct incorrect end repli end sptv
    congrat 1 year special cinema pass 2 call 09061209465 c suprman v matrix3 starwars3 etc 4 free 150pm dont miss
    valu custom pleas advis follow recent review mob award bonu prize call 09066364589
    urgent ur award complimentari trip eurodisinc trav aco entry41 claim txt di 87121 morefrmmob shracomorsglsuplt 10 ls1 3aj
    hear new divorc barbi come ken stuff
    pleas call custom servic repres 0800 169 6031 guarante cash prize
    free rington wait collect simpli text password mix 85069 verifi get usher britney fml po box 5249 mk17 92h 450ppw 16
    gent tri contact last weekend draw show prize guarante call claim code k52 valid 12hr 150ppm
    winner u special select 2 receiv 4 holiday flight inc speak live oper 2 claim
    privat 2004 account statement 07742676969 show 786 unredeem bonu point claim call 08719180248 identifi code 45239 expir
    urgent mobil award bonu caller prize final tri contact u call landlin 09064019788 box42wr29c 150ppm
    today voda number end 7548 select receiv 350 award match pleas call 08712300220 quot claim code 4041 standard rate app
    sunshin quiz wkli q win top soni dvd player u know countri algarv txt ansr 82277 sp tyron
    want 2 get laid tonight want real dog locat sent direct 2 ur mob join uk largest dog network bt txting gravel 69888 nt ec2a 150p
    rcv msg chat svc free hardcor servic text go 69988 u get noth u must age verifi yr network tri
    freemsg repli text randi sexi femal live local luv hear netcollex ltd 08700621170150p per msg repli stop end
    custom servic annonc new year deliveri wait pleas call 07046744435 arrang deliveri
    winner u special select 2 receiv cash 4 holiday flight inc speak live oper 2 claim 0871277810810
    stop bootydeli invit friend repli see stop send stop frnd 62468
    bangbab ur order way u receiv servic msg 2 download ur content u goto wap bangb tv ur mobil menu
    urgent tri contact last weekend draw show prize guarante call claim code s89 valid 12hr
    pleas call custom servic repres freephon 0808 145 4742 guarante cash prize
    uniqu enough find 30th august
    500 new mobil 2004 must go txt nokia 89545 collect today 2optout
    u meet ur dream partner soon ur career 2 flyng start 2 find free txt horo follow ur star sign horo ari
    text meet someon sexi today u find date even flirt join 4 10p repli name age eg sam 25 18 recd thirtyeight penc
    u 447801259231 secret admir look 2 make contact r reveal think ur 09058094597
    congratul ur award 500 cd voucher 125gift guarante free entri 2 100 wkli draw txt music 87066 tnc
    tri contact repli offer video handset 750 anytim network min unlimit text camcord repli call 08000930705
    hey realli horni want chat see nake text hot 69698 text charg 150pm unsubscrib text stop 69698
    ur rington servic chang 25 free credit go choos content stop txt club stop 87070 club4 po box1146 mk45 2wt
    rington club get uk singl chart mobil week choos top qualiti rington messag free charg
    hmv bonu special 500 pound genuin hmv voucher answer 4 easi question play send hmv 86688 info
    custom may claim free camera phone upgrad pay go sim card loyalti call 0845 021 end c appli
    sm ac blind date 4u rodds1 aberdeen unit kingdom check http sm blind date send hide
    themob check newest select content game tone gossip babe sport keep mobil fit funki text wap 82468
    think ur smart win week weekli quiz text play 85222 cs winnersclub po box 84 m26 3uz
    decemb mobil entitl updat latest colour camera mobil free call mobil updat co free 08002986906
    call germani 1 penc per minut call fix line via access number 0844 861 85 prepay direct access
    valentin day special win quiz take partner trip lifetim send go 83600 rcvd
    fanci shag txt xxuk suzi txt cost per msg tnc websit x
    ur current 500 pound maxim ur send cash 86688 cc 08708800282
    xma offer latest motorola sonyericsson nokia free bluetooth doubl min 1000 txt orang call mobileupd8 08000839402
    discount code rp176781 stop messag repli stop custom servic 08717205546
    thank rington order refer t91 charg gbp 4 per week unsubscrib anytim call custom servic 09057039994
    doubl min txt 4 6month free bluetooth orang avail soni nokia motorola phone call mobileupd8 08000839402
    4mth half price orang line rental latest camera phone 4 free phone 11mth call mobilesdirect free 08000938767 updat or2stoptxt
    free rington text first 87131 poli text get 87131 true tone help 0845 2814032 16 1st free tone txt stop
    100 date servic cal l 09064012103 box334sk38ch
    free entri weekli competit text word win 80086 18 c
    send logo 2 ur lover 2 name join heart txt love name1 name2 mobno eg love adam eve 07123456789 87077 yahoo pobox36504w45wq txtno 4 ad 150p
    someon contact date servic enter phone fanci find call landlin 09111032124 pobox12n146tf150p
    urgent mobil number award prize guarante call 09058094455 land line claim valid 12hr
    congrat nokia 3650 video camera phone call 09066382422 call cost 150ppm ave call 3min vari mobil close 300603 post bcm4284 ldn wc1n3xx
    loan purpos homeown tenant welcom previous refus still help call free 0800 1956669 text back
    upgrdcentr orang custom may claim free camera phone upgrad loyalti call 0207 153 offer end 26th juli c appli avail
    okmail dear dave final notic collect 4 tenerif holiday 5000 cash award call 09061743806 landlin tc sae box326 cw25wx 150ppm
    want 2 get laid tonight want real dog locat sent direct 2 ur mob join uk largest dog network txting moan 69888nyt ec2a 150p
    free messag activ 500 free text messag repli messag word free term condit visit
    congratul week competit draw u prize claim call 09050002311 b4280703 sm 18 150ppm
    guarante latest nokia phone 40gb ipod mp3 player prize txt word collect 83355 ibhltd ldnw15h
    boltblu tone 150p repli poli mono eg poly3 cha cha slide yeah slow jamz toxic come stop 4 tone txt
    credit top http renew pin tgxxrz
    urgent mobil award bonu caller prize 2nd attempt contact call box95qu
    today offer claim ur worth discount voucher text ye 85023 savamob member offer mobil cs 08717898035 sub 16 unsub repli x
    reciev tone within next 24hr term condit pleas see channel u teletext pg 750
    privat 2003 account statement 07815296484 show 800 point call 08718738001 identifi code 41782 expir
    monthlysubscript csc web age16 2stop txt stop
    cash prize claim call09050000327
    mobil number claim call us back ring claim hot line 09050005321
    tri contact repli offer 750 min 150 textand new video phone call 08002988890 repli free deliveri tomorrow
    ur chanc win wkli shop spree txt shop c custcar 08715705022
    special select receiv 2000 pound award call 08712402050 line close cost 10ppm cs appli ag promo
    privat 2003 account statement 07753741225 show 800 point call 08715203677 identifi code 42478 expir
    import custom servic announc call freephon 0800 542 0825
    xclusiv clubsaisai 2morow soire special zouk nichol rose 2 ladi info
    22 day kick euro2004 u kept date latest news result daili remov send get txt stop 83222
    new textbuddi chat 2 horni guy ur area 4 25p free 2 receiv search postcod txt one name 89693
    today vodafon number end 4882 select receiv award number match call 09064019014 receiv award
    dear voucher holder 2 claim week offer pc go http ts cs stop text txt stop 80062
    privat 2003 account statement show 800 point call 08715203694 identifi code 40533 expir
    cash prize claim call09050000327 c rstm sw7 3ss 150ppm
    88800 89034 premium phone servic call 08718711108
    sm ac sun0819 post hello seem cool want say hi hi stop send stop 62468
    get ur 1st rington free repli msg tone gr8 top 20 tone phone everi week per wk 2 opt send stop 08452810071 16
    hi sue 20 year old work lapdanc love sex text live bedroom text sue textoper g2 1da 150ppmsg
    forward 448712404000 pleas call 08712404000 immedi urgent messag wait
    review keep fantast nokia game deck club nokia go 2 unsubscrib alert repli word
    4mth half price orang line rental latest camera phone 4 free phone call mobilesdirect free 08000938767 updat or2stoptxt cs
    08714712388 cost 10p
    urgent 2nd attempt contact u u call 09071512433 b4 050703 csbcm4235wc1n3xx callcost 150ppm mobilesvari 50
    guarante cash prize claim yr prize call custom servic repres 08714712394
    email alertfrom jeri stewarts 2kbsubject prescripiton drvgsto listen email call 123
    hi custom loyalti offer new nokia6650 mobil txtauction txt word start 81151 get 4t ctxt tc
    u subscrib best mobil content servic uk per 10 day send stop helplin 08706091795
    realiz 40 year thousand old ladi run around tattoo
    import custom servic announc premier
    romant pari 2 night 2 flight book 4 next year call 08704439680t cs appli
    urgent ur guarante award still unclaim call 09066368327 claimcod m39m51
    ur award citi break could win summer shop spree everi wk txt store 88039 skilgm tscs087147403231winawk age16
    import custom servic announc premier call freephon 0800 542 0578
    ever thought live good life perfect partner txt back name age join mobil commun
    5 free top polyphon tone call 087018728737 nation rate get toppoli tune sent everi week text subpoli 81618 per pole unsub 08718727870
    orang custom may claim free camera phone upgrad loyalti call 0207 153 offer end 14thmarch c appli availa
    last chanc claim ur worth discount voucher today text shop 85023 savamob offer mobil cs savamob pobox84 m263uz sub 16
    free 1st week no1 nokia tone 4 ur mobil everi week txt nokia 8077 get txting tell ur mate pobox 36504 w45wq
    guarante award even cashto claim ur award call free 08000407165 2 stop getstop 88222 php rg21 4jx
    congratul ur award either cd gift voucher free entri 2 weekli draw txt music 87066 tnc
    u outbid simonwatson5120 shinco dvd plyr 2 bid visit sm 2 end bid notif repli end
    smsservic yourinclus text credit pl goto 3qxj9 unsubscrib stop extra charg help 9ae
    25p 4 alfi moon children need song ur mob tell ur m8 txt tone chariti 8007 nokia poli chariti poli zed 08701417012 profit 2 chariti
    u secret admir reveal think u r special call opt repli reveal stop per msg recd cust care 07821230901
    dear voucher holder claim week offer pc pleas go http ts cs appli stop text txt stop 80062
    want 750 anytim network min 150 text new video phone five pound per week call 08002888812 repli deliveri tomorrow
    tri contact offer new video phone 750 anytim network min half price rental camcord call 08000930705 repli deliveri wed
    last chanc 2 claim ur worth discount ye 85023 offer mobil cs 08717898035 sub 16 remov txt x stop
    urgent call 09066350750 landlin complimentari 4 ibiza holiday cash await collect sae cs po box 434 sk3 8wp 150 ppm
    talk sexi make new friend fall love world discreet text date servic text vip 83110 see could meet
    congratul ur award either yr suppli cd virgin record mysteri gift guarante call 09061104283 ts cs approx 3min
    privat 2003 account statement 07808 xxxxxx show 800 point call 08719899217 identifi code 41685 expir
    hello need posh bird chap user trial prod champney put need address dob asap ta r
    u want xma 100 free text messag new video phone half price line rental call free 0800 0721072 find
    shop till u drop either 10k 5k cash travel voucher call ntt po box cr01327bt fixedlin cost 150ppm mobil vari
    sunshin quiz wkli q win top soni dvd player u know countri liverpool play mid week txt ansr 82277 sp tyron
    u secret admir look 2 make contact r reveal think ur 09058094565
    u secret admir look 2 make contact r reveal think ur
    remind download content alreadi paid goto http mymobi collect content
    lastest stereophon marley dizze racal libertin stroke win nookii game flirt click themob wap bookmark text wap 82468
    januari male sale hot gay chat cheaper call nation rate cheap peak stop text call 08712460324
    money r lucki winner 2 claim prize text money 2 88600 give away text rate box403 w1t1ji
    dear matthew pleas call 09063440451 landlin complimentari 4 lux tenerif holiday cash await collect ppm150 sae cs box334 sk38xh
    urgent call 09061749602 landlin complimentari 4 tenerif holiday cash await collect sae cs box 528 hp20 1yf 150ppm
    get touch folk wait compani txt back name age opt enjoy commun
    ur current 500 pound maxim ur send go 86688 cc 08718720201 po box
    filthi stori girl wait
    urgent tri contact today draw show prize guarante call 09050001808 land line claim m95 valid12hr
    congrat 2 mobil 3g videophon r call 09063458130 videochat wid mate play java game dload polyph music nolin rentl
    panason bluetoothhdset free nokia free motorola free doublemin doubletxt orang contract call mobileupd8 08000839402 call 2optout
    free 1st week no1 nokia tone 4 ur mob everi week txt nokia 8007 get txting tell ur mate pobox 36504 w45wq
    guess somebodi know secretli fanci wan na find give us call 09065394514 landlin datebox1282essexcm61xn 18
    know someon know fanci call 09058097218 find pobox 6 ls15hb 150p
    1000 flirt txt girl bloke ur name age eg girl zoe 18 8007 join get chat
    18 day euro2004 kickoff u kept inform latest news result daili unsubscrib send get euro stop 83222
    eastend tv quiz flower dot compar violet tulip lili txt e f 84025 4 chanc 2 win cash
    new local date area lot new peopl regist area repli date start 18 replys150
    someon u know ask date servic 2 contact cant guess call 09058091854 reveal po box385 m6 6wu
    urgent tri contact today draw show prize guarante call 09050003091 land line claim c52 valid12hr
    dear u invit xchat final attempt contact u txt chat 86688
    award sipix digit camera call 09061221061 landlin deliveri within 28day cs box177 m221bp 2yr warranti 150ppm 16 p
    win urgent mobil number award prize guarante call 09061790121 land line claim 3030 valid 12hr 150ppm
    dear subscrib ur draw 4 gift voucher b enter receipt correct an elvi presley birthday txt answer 80062
    messag import inform o2 user today lucki day 2 find log onto http fantast surpris await
    449050000301 price claim call 09050000301
    bore speed date tri speedchat txt speedchat 80155 like em txt swap get new chatter chat80155 pobox36504w45wq rcd 16
    want 750 anytim network min 150 text new video phone five pound per week call 08000776320 repli deliveri tomorrow
    take part mobil survey yesterday 500 text 2 use howev wish 2 get txt send txt 80160 c
    ur hmv quiz current maxim ur send hmv1 86688
    dont forget place mani free request wish inform call 08707808226
    know u u know send chat 86688 let find rcvd ldn 18 year
    thank winner notifi sm good luck futur market repli stop 84122 custom servic 08450542832
    1000 girl mani local 2 u r virgin 2 r readi 2 4fil ur everi sexual need u 4fil text cute 69911
    got take 2 take part wrc ralli oz u lucozad energi text ralli le 61200 25p see pack itcould u
    sex ur mobil free sexi pic jordan text babe everi wk get sexi celeb 4 pic 16 087016248
    1 new voicemail pleas call 08719181503
    win year suppli cd 4 store ur choic worth enter weekli draw txt music 87066 ts cs
    sim subscrib select receiv bonu get deliv door txt word ok 88600 claim exp 30apr
    1 new voicemail pleas call 08719181513
    1 nokia tone 4 ur mob everi week txt nok 87021 1st tone free get txtin tell ur friend 16 repli hl 4info
    repli name address receiv post week complet free accommod variou global locat
    free entri weekli comp send word enter 84128 18 c cust care 08712405020
    pleas call 08712402779 immedi urgent messag wait
    hungri gay guy feel hungri 4 call 08718730555 stop text call 08712460324
    u get 2 phone wan na chat 2 set meet call 09096102316 u cum 2moro luv jane xx
    network oper servic free c visit
    enjoy jamster videosound gold club credit 2 new get fun help call 09701213186
    get 3 lion england tone repli lionm 4 mono lionp 4 poli 4 go 2 origin n best tone 3gbp network oper rate appli
    win newest harri potter order phoenix book 5 repli harri answer 5 question chanc first among reader
    ur balanc ur next question sang girl 80 2 answer txt ur answer good luck
    free2day sexi st georg day pic jordan txt pic 89080 dont miss everi wk sauci celeb 4 pic c 0870241182716
    hot live fantasi call 08707509020 20p per min ntt ltd po box 1327 croydon cr9 5wb 0870 k
    bear pic nick tom pete dick fact type tri gay chat photo upload call 08718730666 2 stop text call 08712460324
    500 new mobil 2004 must go txt nokia 89545 collect today 2optout txtauction
    doubl min doubl txt price linerent latest orang bluetooth mobil call mobileupd8 latest offer 08000839402
    urgent import inform o2 user today lucki day 2 find log onto http fantast surpris await
    dear u invit xchat final attempt contact u txt chat 86688 ldn 18 yr
    congratul ur award either cd gift voucher free entri 2 weekli draw txt music 87066 tnc 1 win150ppmx3age16
    sale arsen dartboard good condit doubl trebl
    free 1st week entri 2 textpod 4 chanc 2 win 40gb ipod cash everi wk txt pod 84128 ts cs custcar 08712405020
    regist optin subscrib ur draw 4 gift voucher enter receipt correct an 80062 what no1 bbc chart
    summer final fanci chat flirt sexi singl yr area get match repli summer free 2 join optout txt stop help08714742804
    clair havin borin time alon u wan na cum 2nite chat 09099725823 hope 2 c u luv clair xx
    bought one rington get text cost 3 pound offer tone etc
    09066362231 urgent mobil 07xxxxxxxxx bonu caller prize 2nd attempt reach call 09066362231 asap
    07801543489 guarante latest nokia phone 40gb ipod mp3 player prize txt word collect
    hi luci hubbi meetin day fri b alon hotel u fanci cumin pl leav msg 2day 09099726395 luci x
    account credit 500 free text messag activ txt word credit 80488 cs
    sm ac jsco energi high u may know 2channel 2day ur leadership skill r strong psychic repli an end repli end jsco
    hot live fantasi call 08707509020 20p per min ntt ltd po box 1327 croydon cr9 5wb 0870 nation rate call
    thank vote sing along star karaok mobil free link repli sing
    brand new mobil music servic live free music player arriv shortli instal phone brows content top artist
    urgent mobil award bonu caller prize 2nd attempt contact call box95qu bt nation rate
    nokia 7250i get win free auction take part send nokia 86021
    hello orang 1 month free access game news sport plu 10 free text 20 photo messag repli ye term appli
    ur current 500 pound maxim ur send go 86688 cc 08718720201
    sm auction brand new nokia 7250 4 auction today auction free 2 join take part txt nokia 86021
    privat 2003 account statement show 800 point call 08719899230 identifi code 41685 expir
    regist subscrib yr draw 4 gift voucher b enter receipt correct an next olymp txt an 80062
    urgent mobil number award prize guarante call 09061790121 land line claim valid 12hr 150ppm
    pro video club need help info call 08701237397 must club credit redeem enjoy
    u secret admir look 2 make contact r reveal think ur 09058094599
    500 free text msg text ok 80488 credit account
    select stay 1 250 top british hotel noth holiday worth claim call london bx 526 sw73ss
    eeri nokia tone 4u rpli tone titl 8007 eg tone dracula 8007 titl ghost addamsfa munster exorcist twilight pobox36504w45wq 150p
    0a network allow compani bill sm respons supplier shop give guarante sell g
    freemsg feelin kinda lnli hope u like 2 keep compani jst got cam mobi wan na c pic txt repli date 82242 msg150p 2rcv hlp 08712317606 stop 82242
    ur chanc win cash everi wk txt action c custcar 08712405022
    rgent 2nd attempt contact u u call 09071512433 b4 050703 csbcm4235wc1n3xx callcost 150ppm mobilesvari 50
    hi ur lookin 4 sauci daytim fun wiv busti marri woman free next week chat 2 sort time 09099726429 janinexx
    urgent tri contact today draw show prize guarante call 09050001295 land line claim a21 valid 12hr
    monthli password wap use wap phone pc
    today vodafon number end 0089 last four digit select receiv award number match pleas call 09063442151 claim award
    free top rington weekli 1st week subpoli 3 per
    free msg sorri servic order 81303 could deliv suffici credit pleas top receiv servic
    hard live 121 chat choos girl connect live call 09094646899 cheap chat uk biggest live servic vu bcm1896wc1n3xx
    wow boy r back take 2007 uk tour win vip ticket vip club txt club trackmarqu ltd info vipclub4u
    hi mandi sullivan call hotmix fm chosen receiv easter prize draw pleas telephon 09041940223 claim prize transfer someon els
    ur go 2 bahama callfreefon 08081560665 speak live oper claim either bahama cruis cash opt txt x 07786200117
    someon conact date servic enter phone fanci find call landlin pobox12n146tf15
    hi 07734396839 ibh custom loyalti offer new nokia6600 mobil txtauction txt word start get 4t
    sm auction nokia 7250i get win free auction take part send nokia 86021
    call freephon 0800 542 0578
    buy space invad 4 chanc 2 win orig arcad game consol press 0 game arcad std wap charg see 4 term set purchas
    big brother alert comput select u 10k cash 150 voucher call ntt po box cro1327 bt landlin cost 150ppm mobil vari
    win winner foley ipod excit prize soon keep eye ur mobil visit
    today voda number end 1225 select receiv match pleas call 08712300220 quot claim code 3100 standard rate app
    hottest pic straight phone see get wet want xx text pic 89555 txt cost 150p textoper g696ga 18 xxx
    hack chat get backdoor entri 121 chat room fraction cost repli neo69 call 09050280520 subscrib 25p pm dp bcm box 8027 ldn wc1n3xx
    free nokia motorola upto 12mth linerent 500 free min free call mobileupd8 08001950382 call
    2nd time tri 2 contact u 750 pound prize 2 claim easi call 08718726970 10p per min
    guarante cash claim yr prize call custom servic repres
    would like see xxx pic hot nearli ban uk
    u secret admir look 2 make contact r reveal think ur 09058094594
    dear 0776xxxxxxx u invit xchat final attempt contact u txt chat 86688 ldn 18yr
    urgent pleas call 09061743811 landlin abta complimentari 4 tenerif holiday cash await collect sae cs box 326 cw25wx 150ppm
    call 09090900040 listen extrem dirti live chat go offic right total privaci one know sic listen 60p min
    freemsg hey u got 1 fone repli wild txt ill send u pic hurri im bore work xxx 18 stop2stop
    free entri 2 weekli comp chanc win ipod txt pod 80182 get entri std txt rate c appli 08452810073 detail
    new textbuddi chat 2 horni guy ur area 4 25p free 2 receiv search postcod txt one name 89693 08715500022 rpl stop 2 cnl
    call 08702490080 tell u 2 call 09066358152 claim prize u 2 enter ur mobil person detail prompt care
    free 1st week entri 2 textpod 4 chanc 2 win 40gb ipod cash everi wk txt vpod 81303 ts cs custcar 08712405020
    peopl dog area call 09090204448 join like mind guy arrang 1 1 even minapn ls278bb
    well done 4 costa del sol holiday await collect call 09050090044 toclaim sae tc pobox334 stockport sk38xh max10min
    guess somebodi know secretli fanci wan na find give us call 09065394973 landlin datebox1282essexcm61xn 18
    500 free text messag valid 31 decemb 2005
    guarante award even cashto claim ur award call free 08000407165 2 stop getstop 88222 php
    repli win weekli 2006 fifa world cup held send stop 87239 end servic
    urgent pleas call 09061743810 landlin abta complimentari 4 tenerif holiday 5000 cash await collect sae cs box 326 cw25wx 150 ppm
    free tone hope enjoy new content text stop 61610 unsubscrib provid
    themob yo yo come new select hot download member get free click open next link sent ur fone
    great news call freefon 08006344447 claim guarante cash gift speak live oper
    u win music gift voucher everi week start txt word draw 87066 tsc
    call 09094100151 use ur min call cast mob vari servic provid aom aom box61 m60 1er u stop age
    urgent mobil bonu caller prize 2nd attempt reach call 09066362220 asap box97n7qp 150ppm
    sexi singl wait text age follow gender wither f gay men text age follow
    freemsg claim ur 250 sm ok 84025 use web2mobil 2 ur mate etc join c box139 la32wu 16 remov txtx stop
    85233 free rington repli real
    well done england get offici poli rington colour flag yer mobil text tone flag 84199 txt eng stop box39822 w111wx
    final chanc claim ur worth discount voucher today text ye 85023 savamob member offer mobil cs savamob pobox84 m263uz sub 16
    sm servic inclus text credit pl goto unsubscrib stop extra charg po box420 ip4 5we
    winner special select receiv cash award speak live oper claim call cost 10p
    sunshin hol claim ur med holiday send stamp self address envelop drink us uk po box 113 bray wicklow eir quiz start saturday unsub stop
    u win music gift voucher everi week start txt word draw 87066 tsc skillgam 1winaweek age16 150ppermesssubscript
    b4u voucher marsm log onto discount credit opt repli stop custom care call 08717168528
    freemsg hey buffi 25 love satisfi men home alon feel randi repli 2 c pix qlynnbv help08700621170150p msg send stop stop txt
    free 1st week no1 nokia tone 4 ur mob everi week txt nokia 87077 get txting tell ur mate zed pobox 36504 w45wq
    free camera phone linerent 750 cross ntwk min price txt bundl deal also avbl call 08001950382 mf
    urgent mobil 07xxxxxxxxx bonu caller prize 2nd attempt reach call 09066362231 asap box97n7qp 150ppm
    urgent 4 costa del sol holiday await collect call 09050090044 toclaim sae tc pobox334 stockport sk38xh max10min
    guarante cash prize claim yr prize call custom servic repres 08714712379 cost 10p
    thank rington order ref number k718 mobil charg tone arriv pleas call custom servic 09065069120
    hi ya babe x u 4goten bout scammer get smart though regular vodafon respond get prem rate no use also bewar
    back 2 work 2morro half term u c 2nite 4 sexi passion b4 2 go back chat 09099726481 luv dena call
    thank rington order ref number r836 mobil charg tone arriv pleas call custom servic 09065069154
    splashmobil choos 1000 gr8 tone wk subscrit servic weekli tone cost 300p u one credit kick back enjoy
    heard u4 call 4 rude chat privat line 01223585334 cum wan 2c pic gettin shag text pix 8552 2end send stop 8552 sam xxx
    forward 88877 free entri weekli comp send word enter 88877 18 c
    88066 88066 lost 3pound help
    mobil 11mth updat free orang latest colour camera mobil unlimit weekend call call mobil upd8 freefon 08000839402 2stoptx
    1 new messag pleas call 08718738034
    forward 21870000 hi mailbox messag sm alert 4 messag 21 match pleas call back 09056242159 retriev messag match
    mobi pub high street prize u know new duchess cornwal txt first name stop 008704050406 sp arrow
    congratul thank good friend u xma prize 2 claim easi call 08718726971 10p per minut
    tddnewslett game thedailydraw dear helen dozen free game great prizeswith
    urgent mobil number bonu caller prize 2nd attempt reach call 09066368753 asap box 97n7qp 150ppm
    doubl min txt orang price linerent motorola sonyericsson free call mobileupd8 08000839402
    download mani rington u like restrict 1000 2 choos u even send 2 yr buddi txt sir 80082
    pleas call 08712402902 immedi urgent messag wait
    spook mob halloween collect logo pic messag plu free eeri tone txt card spook 8007 zed 08701417012150p per
    fantasi footbal back tv go sky gamestar sky activ play dream team score start saturday regist sky opt 88088
    tone club sub expir 2 repli monoc 4 mono polyc 4 poli 1 weekli 150p per week txt stop 2 stop msg free stream 0871212025016
    xma prize draw tri contact today draw show prize guarante call 09058094565 land line valid 12hr
    ye place town meet excit adult singl uk txt chat 86688
    someon contact date servic enter phone becausethey fanci find call landlin pobox1 w14rg 150p
    babe u want dont u babi im nasti thing 4 filthyguy fanci rude time sexi bitch go slo n hard txt xxx slo 4msg
    sm servic inclus text credit pl gotto login 3qxj9 unsubscrib stop extra charg help 08702840625 9ae
    valentin day special win quiz take partner trip lifetim send go 83600 rcvd
    guess first time creat web page read wrote wait opinion want friend
    ur chanc win cash everi wk txt play c custcar 08715705022
    sppok ur mob halloween collect nokia logo pic messag plu free eeri tone txt card spook 8007
    urgent call 09066612661 landlin complementari 4 tenerif holiday cash await collect sae cs po box 3 wa14 2px 150ppm sender hol offer
    winner valu network custom hvae select receiv reward collect call valid 24 hour acl03530150pm
    u nokia 6230 plu free digit camera u get u win free auction take part send nokia 83383 16
    free entri weekli comp send word win 80086 18 c
    text82228 get rington logo game question info
    freemsg award free mini digit camera repli snap collect prize quizclub opt stop sp rwm
    messag brought gmw connect
    congrat 2 mobil 3g videophon r call 09063458130 videochat wid ur mate play java game dload polyph music nolin rentl bx420 ip4 5we 150p
    next amaz xxx picsfree1 video sent enjoy one vid enough 2day text back keyword picsfree1 get next video
    u subscrib best mobil content servic uk per ten day send stop helplin 08706091795
    3 free tarot text find love life tri 3 free text chanc 85555 16 3 free msg
    join uk horniest dog servic u sex 2nite sign follow instruct txt entri 69888 150p
    knock knock txt whose 80082 enter r weekli draw 4 gift voucher 4 store yr choic cs age16
    forward 21870000 hi mailbox messag sm alert 40 match pleas call back 09056242159 retriev messag match
    free ring tone text poli everi week get new tone 0870737910216yr
    urgent mobil 077xxx bonu caller prize 2nd attempt reach call 09066362206 asap box97n7qp 150ppm
    guarante latest nokia phone 40gb ipod mp3 player prize txt word collect 83355 ibhltd ldnw15h
    hello darl today would love chat dont tell look like sexi
    8007 free 1st week no1 nokia tone 4 ur mob everi week txt nokia 8007 get txting tell ur mate pobox 36504 w4 5wq norm
    wan na get laid 2nite want real dog locat sent direct ur mobil join uk largest dog network txt park 69696 nyt ec2a 3lp
    tri contact respons offer new nokia fone camcord hit repli call 08000930705 deliveri
    new tone week includ 1 ab 2 sara 3 order follow instruct next messag
    urgent tri contact today draw show prize guarante call 09050003091 land line claim c52 valid 12hr
    sport fan get latest sport news str 2 ur mobil 1 wk free plu free tone txt sport 8007 norm
    urgent urgent 800 free flight europ give away call b4 10th sept take friend 4 free call claim ba128nnfwfly150ppm
    88066 lost help
    freemsg fanci flirt repli date join uk fastest grow mobil date servic msg rcvd 25p optout txt stop repli date
    great new offer doubl min doubl txt best orang tariff get latest camera phone 4 free call mobileupd8 free 08000839402 2stoptxt cs
    hope enjoy new content text stop 61610 unsubscrib provid
    urgent pleas call 09066612661 landlin cash luxuri 4 canari island holiday await collect cs sae award 20m12aq 150ppm
    urgent pleas call 09066612661 landlin complimentari 4 lux costa del sol holiday cash await collect ppm 150 sae cs jame 28 eh74rr
    marri local women look discreet action 5 real match instantli phone text match 69969 msg cost 150p 2 stop txt stop bcmsfwc1n3xx
    burger king wan na play footi top stadium get 2 burger king 1st sept go larg super walk winner
    come take littl time child afraid dark becom teenag want stay night
    ur chanc win cash everi wk txt action c custcar 08712405022
    u bin award play 4 instant cash call 08715203028 claim everi 9th player win min optout 08718727870
    freemsg fav xma tone repli real
    gr8 poli tone 4 mob direct 2u rpli poli titl 8007 eg poli breathe1 titl crazyin sleepingwith finest ymca pobox365o4w45wq 300p
    interflora late order interflora flower christma call 0800 505060 place order midnight tomorrow
    romcapspam everyon around respond well presenc sinc warm outgo bring real breath sunshin
    congratul thank good friend u xma prize 2 claim easi call 08712103738 10p per minut
    send logo 2 ur lover 2 name join heart txt love name1 name2 mobno eg love adam eve 07123456789 87077 yahoo pobox36504w45wq txtno 4 ad 150p
    tkt euro2004 cup final cash collect call 09058099801 b4190604 pobox 7876150ppm
    jamster get crazi frog sound poli text mad1 real text mad2 88888 6 crazi sound 3 c appli
    chanc realiti fantasi show call 08707509020 20p per min ntt ltd po box 1327 croydon cr9 5wb 0870 nation rate call
    adult 18 content video shortli
    chanc realiti fantasi show call 08707509020 20p per min ntt ltd po box 1327 croydon cr9 5wb 0870 nation rate call
    hey boy want hot xxx pic sent direct 2 ur phone txt porn 69855 24hr free 50p per day stop text stopbcm sf wc1n3xx
    doubl min 1000 txt orang tariff latest motorola sonyericsson nokia bluetooth free call mobileupd8 08000839402 yhl
    ur current 500 pound maxim ur send cash 86688 cc 08718720201 po box
    urgent mobil number award prize guarante call 09058094454 land line claim valid 12hr
    sorri u unsubscrib yet mob offer packag min term 54 week pl resubmit request expiri repli themob help 4 info
    1 new messag pleas call 08712400200
    current messag await collect collect messag call 08718723815
    urgent mobil award bonu caller prize final attempt 2 contact u call 08714714011
    ever notic drive anyon go slower idiot everyon drive faster maniac
    xma offer latest motorola sonyericsson nokia free bluetooth dvd doubl min 1000 txt orang call mobileupd8 08000839402
    repli win weekli profession sport tiger wood play send stop 87239 end servic
    1 polyphon tone 4 ur mob everi week txt pt2 87575 1st tone free get txtin tell ur friend 16 repli hl 4info
    messag free welcom new improv sex dog club unsubscrib servic repli stop msg 150p
    12mth half price orang line rental 400min call mobileupd8 08000839402
    free unlimit hardcor porn direct 2 mobil txt porn 69200 get free access 24 hr chrgd 50p per day txt stop 2exit msg free
    unsubscrib servic get ton sexi babe hunk straight phone go http subscript
    hi babe jordan r u im home abroad lone text back u wan na chat xxsp text stop stopcost 150p 08712400603
    get brand new mobil phone agent mob plu load goodi info text mat 87021
    lord ring return king store repli lotr 2 june 4 chanc 2 win lotr soundtrack cd stdtxtrate repli stop end txt
    good luck draw take place 28th feb good luck remov send stop 87239 custom servic 08708034412
    1st wk free gr8 tone str8 2 u wk txt nokia 8007 classic nokia tone hit 8007 poli
    lookatm thank purchas video clip lookatm charg 35p think better send video mmsto 32323
    sexi sexi cum text im wet warm readi porn u fun msg free recd msg 150p inc vat 2 cancel text stop
    2nd time tri contact u prize claim call 09053750005 b4 sm 08718725756 140ppm
    dear voucher holder claim week offer pc pleas go http ts cs appli
    2nd time tri 2 contact u 750 pound prize 2 claim easi call 08712101358 10p per min
    ur award citi break could win summer shop spree everi wk txt store
    urgent tri contact today draw show prize guarante call 09066358361 land line claim y87 valid 12hr
    thank rington order refer number x29 mobil charg tone arriv pleas call custom servic 09065989180
    ur current 500 pound maxim ur send collect 83600 cc 08718720201 po box
    congratul thank good friend u xma prize 2 claim easi call 08718726978 10p per minut
    44 7732584351 want new nokia 3510i colour phone deliveredtomorrow 300 free minut mobil 100 free text free camcord repli call 08000930705
    someon u know ask date servic 2 contact cant guess call 09058097189 reveal pobox 6 ls15hb 150p
    camera award sipix digit camera call 09061221066 fromm landlin deliveri within 28 day
    today voda number end 5226 select receiv 350 award hava match pleas call 08712300220 quot claim code 1131 standard rate app
    messag free welcom new improv sex dog club unsubscrib servic repli stop msg 150p 18
    rct thnq adrian u text rgd vatian
    contact date servic someon know find call land line pobox45w2tg150p
    sorri miss call let talk time 07090201529
    complimentari 4 star ibiza holiday cash need urgent collect 09066364349 landlin lose
    free msg bill mobil number mistak shortcod call 08081263000 charg call free bt landlin
    pleas call 08712402972 immedi urgent messag wait
    urgent mobil number award bonu caller prize call 09058095201 land line valid 12hr
    want new nokia 3510i colour phone deliveredtomorrow 300 free minut mobil 100 free text free camcord repli call 08000930705
    life never much fun great came made truli special wo forget enjoy one
    want new video phone 600 anytim network min 400 inclus video call download 5 per week free deltomorrow call 08002888812 repli
    valu custom pleas advis follow recent review mob award bonu prize call 09066368470
    welcom pleas repli age gender begin 24m
    freemsg unlimit free call activ smartcal txt call unlimit call help 08448714184 stop txt stop landlineonli
    mobil 10 mth updat latest orang phone free save free call text ye callback orno opt
    new 2 club dont fink met yet b gr8 2 c u pleas leav msg 2day wiv ur area 09099726553 repli promis carli x lkpobox177hp51fl
    camera award sipix digit camera call 09061221066 fromm landlin deliveri within 28 day
    get free mobil video player free movi collect text go free extra film order c appli 18 yr
    save money wed lingeri choos superb select nation deliveri brought weddingfriend
    heard u4 call night knicker make beg like u last time 01223585236 xx luv
    bloomberg center wait appli futur http
    want new video phone750 anytim network min 150 text five pound per week call 08000776320 repli deliveri tomorrow
    contact date servic someon know find call land line pobox45w2tg150p
    wan2 win westlif 4 u m8 current tour 1 unbreak 2 untam 3 unkempt text 3 cost 50p text
    dorothi bank granit issu explos pick member 300 nasdaq symbol cdgt per
    winner guarante caller prize final attempt contact claim call 09071517866 150ppmpobox10183bhamb64x
    xma new year eve ticket sale club day 10am till 8pm thur fri sat night week sell fast
    rock yr chik get 100 filthi film xxx pic yr phone rpli filth saristar ltd e14 9yt 08701752560 450p per 5 day stop2 cancel
    next month get upto 50 call 4 ur standard network charg 2 activ call 9061100010 c 1st4term pobox84 m26 3uz cost min mobcudb
    urgent tri contact u today draw show prize guarante call 09050000460 land line claim j89 po box245c2150pm
    text banneduk 89555 see cost 150p textoper g696ga xxx
    auction round highest bid next maximum bid bid send bid 10 bid good luck
    collect valentin weekend pari inc flight hotel prize guarante text pari
    custom loyalti offer new nokia6650 mobil txtauction txt word start 81151 get 4t ctxt tc
    wo believ true incred txt repli g learn truli amaz thing blow mind o2fwd
    hot n horni will live local text repli hear strt back 150p per msg netcollex ltdhelpdesk 02085076972 repli stop end
    want new nokia 3510i colour phone deliv tomorrow 200 free minut mobil 100 free text free camcord repli call 08000930705
    congratul winner august prize draw call 09066660100 prize code 2309
    8007 25p 4 alfi moon children need song ur mob tell ur m8 txt tone chariti 8007 nokia poli chariti poli zed 08701417012 profit 2 chariti
    get offici england poli rington colour flag yer mobil tonight game text tone flag optout txt eng stop box39822 w111wx
    custom servic announc recent tri make deliveri unabl pleas call 07090298926
    stop club tone repli stop mix see html term club tone cost mfl po box 1146 mk45 2wt
    wamma get laid want real doggin locat sent direct mobil join uk largest dog network txt dog 69696 nyt ec2a 3lp
    promot number 8714714 ur award citi break could win summer shop spree everi wk txt store 88039 skilgm tscs087147403231winawk age16
    winner special select receiv cash award speak live oper claim call cost 10p
    thank rington order refer number x49 mobil charg tone arriv pleas call custom servic text txtstar
    hi 2night ur lucki night uve invit 2 xchat uk wildest chat txt chat 86688 ldn 18yr
    146tf150p
    dear voucher holder 2 claim 1st class airport loung pass use holiday voucher call book quot 1st class x 2
    someon u know ask date servic 2 contact cant guess call 09058095107 reveal pobox 7 s3xi 150p
    mila age23 blond new uk look sex uk guy u like fun text mtalk 1st 5free increment help08718728876
    claim 200 shop spree call 08717895698 mobstorequiz10ppm
    want funk ur fone weekli new tone repli tones2u 2 text origin n best tone 3gbp network oper rate appli
    twink bear scalli skin jock call miss weekend fun call 08712466669 2 stop text call 08712460324 nat rate
    tri contact repli offer video handset 750 anytim network min unlimit text camcord repli call 08000930705
    urgent tri contact last weekend draw show prize guarante call claim code k61 valid 12hour
    74355 xma iscom ur award either cd gift voucher free entri 2 r weekli draw txt music 87066 tnc
    congratul u claim 2 vip row ticket 2 c blu concert novemb blu gift guarante call 09061104276 claim ts cs
    free msg singl find partner area 1000 real peopl wait chat send chat 62220cncl send stopc per msg
    win newest potter order phoenix book 5 repli harri answer 5 question chanc first among reader
    free msg rington http wml 37819
    oh god found number glad text back xafter msg cst std ntwk chg
    link pictur sent also use http
    doubl min 1000 txt orang tariff latest motorola sonyericsson nokia bluetooth free call mobileupd8 08000839402
    urgent 2nd attempt contact prize yesterday still await collect claim call acl03530150pm
    dear dave final notic collect 4 tenerif holiday 5000 cash award call 09061743806 landlin tc sae box326 cw25wx 150ppm
    tell u 2 call 09066358152 claim prize u 2 enter ur mobil person detail prompt care
    2004 account 07xxxxxxxxx show 786 unredeem point claim call 08719181259 identifi code xxxxx expir
    want new video handset 750 anytim network min half price line rental camcord repli call 08000930705 deliveri tomorrow
    free rington repli real poli eg real1 pushbutton dontcha babygoodby golddigg webeburnin 1st tone free 6 u join
    free msg get gnarl barkley crazi rington total free repli go messag right
    refus loan secur unsecur ca get credit call free 0800 195 6669 text back
    special select receiv 3000 award call 08712402050 line close cost 10ppm cs appli ag promo
    valu vodafon custom comput pick win prize collect easi call 09061743386
    free video camera phone half price line rental 12 mth 500 cross ntwk min 100 txt call mobileupd8 08001950382
    ringtonek 84484
    rington club gr8 new poli direct mobil everi week
    bank granit issu explos pick member 300 nasdaq symbol cdgt per
    bore housew chat n date rate landlin
    tri call repli sm video mobil 750 min unlimit text free camcord repli call 08000930705 del thur
    2nd time tri contact u prize 2 claim easi call 087104711148 10p per minut
    receiv week tripl echo rington shortli enjoy
    u select stay 1 250 top british hotel noth holiday valu dial 08712300220 claim nation rate call bx526 sw73ss
    chosen receiv award pl call claim number 09066364311 collect award select receiv valu mobil custom
    win cash prize prize worth
    thank rington order refer number mobil charg tone arriv pleas call custom servic 09065989182
    mobi pub high street prize u know new duchess cornwal txt first name stop 008704050406 sp
    week savamob member offer access call 08709501522 detail savamob pobox 139 la3 2wu savamob offer mobil
    contact date servic someon know find call mobil landlin 09064017305 pobox75ldns7
    chase us sinc sept definit pay thank inform ignor kath manchest
    loan purpos even bad credit tenant welcom call 08717111821
    87077 kick new season 2wk free goal news ur mobil txt ur club name 87077 eg villa 87077
    orang bring rington time chart hero free hit week go rington pic wap stop receiv tip repli stop
    privat 2003 account statement 07973788240 show 800 point call 08715203649 identifi code 40533 expir
    tri call repli sm video mobil 750 min unlimit text free camcord repli call 08000930705
    gsoh good spam ladi u could b male gigolo 2 join uk fastest grow men club repli oncal mjzgroup repli stop msg
    hot live fantasi call 08707500020 20p per min ntt ltd po box 1327 croydon cr9 5wb 0870 nation rate call
    urgent mobil number award ukp 2000 prize guarante call 09061790125 landlin claim valid 12hr 150ppm
    spjanuari male sale hot gay chat cheaper call nation rate cheap peak stop text call 08712460324
    freemsg today day readi horni live town love sex fun game netcollex ltd 08700621170150p per msg repli stop end
    simpson movi releas juli 2007 name band die start film day day day send b c
    pleas call amanda regard renew upgrad current handset free charg offer end today tel 0845 021 3680 subject c
    want new video phone 750 anytim network min half price line rental free text 3 month repli call 08000930705 free deliveri
    dear voucher holder claim week offer pc pleas go http ts cs appli
    urgent pleas call abta complimentari 4 spanish holiday cash await collect sae cs box 47 po19 2ez 150ppm
    cmon babe make horni turn txt fantasi babe im hot sticki need repli cost 2 cancel send stop
    import inform 4 orang user 0796xxxxxx today ur lucki day 2 find log onto http fantast prizeawait
    miss call alert number call left messag 07008009200
    freemsg record indic may entitl 3750 pound accid claim free repli ye msg opt text stop
    show ur colour euro 2004 offer get england flag 3lion tone ur phone click follow servic messag info
    text pass 69669 collect polyphon rington normal gpr charg appli enjoy tone
    accordingli repeat text word ok mobil phone send
    block breaker come delux format new featur great graphic buy repli get bbdelux take challeng
    import inform 4 orang user today lucki day 2find log onto http fantast surpris await
    natalja invit friend repli see stop send stop frnd 62468
    urgent import inform 02 user today lucki day 2 find log onto http fantast surpris await
    kit strip bill 150p netcollex po box 1013 ig11 oja
    pleas call 08712402578 immedi urgent messag wait
    let send free anonym mask messag im send messag see potenti abus
    congrat 2 mobil 3g videophon r call 09061744553 videochat wid ur mate play java game dload polyh music nolin rentl bx420 ip4 5we 150pm
    import inform 4 orang user 0789xxxxxxx today lucki day 2find log onto http fantast surpris await
    date servic ask 2 contact u someon shi call 09058091870 reveal pobox84 m26 3uz 150p
    want new video handset 750 time network min unlimit text camcord repli call 08000930705 del sat
    ur balanc next question complet landmark big bob barri ben text b c good luck
    ur tonex subscript renew charg choos 10 poli month bill msg
    prize go anoth custom c polo ltd suit 373 london w1j 6hl pleas call back busi
    want new nokia 3510i colour phone deliv tomorrow 200 free minut mobil 100 free text free camcord repli call 8000930705
    recpt order rington order process
    one regist subscrib u enter draw 4 100 gift voucher repli enter unsubscrib text stop
    chanc win free bluetooth headset simpli repli back adp
    b floppi b snappi happi gay chat servic photo upload call 08718730666 2 stop text call 08712460324
    welcom msg free give free call futur mg bill 150p daili cancel send go stop 89123
    receiv mobil content enjoy
    want explicit sex 30 sec ring 02073162414 cost
    latest nokia mobil ipod mp3 player proze guarante repli win 83355 norcorp
    sm servic inclus text credit pl goto 3qxj9 unsubscrib stop extra charg help 9ae
    mobil club choos top qualiti item mobil 7cfca1a
    money wine number 946 wot next
    want cock hubbi away need real man 2 satisfi txt wife 89938 string action txt stop 2 end txt rec otbox 731 la1 7w
    gr8 new servic live sex video chat mob see sexiest dirtiest girl live ur phone 4 detail text horni 89070 cancel send stop 89070
    freemsg hi babi wow got new cam mobi wan na c hot pic fanci chat im w8in 4utxt rpli chat 82242 hlp 08712317606 msg150p 2rcv
    wan na laugh tri mobil logon txting word chat send 8883 cm po box 4217 london w1a 6zf rcvd
    urgent 2nd attempt contact u u 09071512432 b4 300603t 50
    congratul ur award 500 cd voucher 125gift guarante free entri 2 100 wkli draw txt music 87066
    contract mobil 11 mnth latest motorola nokia etc free doubl min text orang tariff text ye callback remov record
    u secret admir look 2 make contact r reveal think ur
    freemsg txt call 86888 claim reward 3 hour talk time use phone inc 3hr 16 stop txtstop
    sunshin quiz win super soni dvd record cannam capit australia text mquiz b
    today voda number end 7634 select receiv reward match pleas call 08712300220 quot claim code 7684 standard rate appli
    rip get mobil content call 08717509990 six download 3
    tri contact repli offer video phone 750 anytim network min half price line rental camcord repli call 08000930705
    xma reward wait comput randomli pick loyal mobil custom receiv reward call 09066380611
    privat 2003 account statement show 800 point call 08718738002 identifi code 48922 expir
    custom servic announc recent tri make deliveri unabl pleas call 07099833605
    hi babe chloe r u smash saturday night great weekend u miss sp text stop stop
    urgent mobil 07808726822 award bonu caller prize 2nd attempt contact call box95qu
    free game get rayman golf 4 free o2 game arcad 1st get ur game set repli post save activ8 press 0 key arcad termsappli
    mobil 10 mth updat latest phone free keep ur number get extra free text ye call
    weekli tone readi download week new tone includ 1 crazi f 2 3 black p info n
    get lot cash weekend dear welcom weekend got biggest best ever cash give away
    thank 4 continu support question week enter u in2 draw 4 cash name new us presid txt an 80082
    uniqu user id remov send stop 87239 custom servic 08708034412
    urgent 09066649731from landlin complimentari 4 ibiza holiday cash await collect sae cs po box 434 sk3 8wp 150ppm
    urgent 2nd attempt contact prize yesterday still await collect claim call 09061702893
    santa call would littl one like call santa xma eve call 09077818151 book time last 3min 30 c
    privat 2004 account statement 078498 7 show 786 unredeem bonu point claim call 08719180219 identifi code 45239 expir
    check choos babe video fgkslpopw fgkslpo
    u r winner u ave special select 2 receiv cash 4 holiday flight inc speak live oper 2 claim 18
    new mobil 2004 must go txt nokia 89545 collect today 2optout txtauction
    privat 2003 account statement show 800 point call 08715203652 identifi code 42810 expir
    free messag thank use auction subscript servic 18 2 skip auction txt 2 unsubscrib txt stop customercar 08718726270
    lyricalladi invit friend repli see stop send stop frnd 62468
    want latest video handset 750 anytim network min half price line rental repli call 08000930705 deliveri tomorrow
    ou guarante latest nokia phone 40gb ipod mp3 player prize txt word collect 83355 ibhltd ldnw15h
    free polyphon rington text super 87131 get free poli tone week 16 sn pobox202 nr31 7z subscript 450pw
    warner villag 83118 c colin farrel swat wkend warner villag get 1 free med popcorn show c c kiosk repli soni 4 mre film offer
    goal arsen 4 henri 7 v liverpool 2 henri score simpl shot 6 yard pass bergkamp give arsen 2 goal margin 78 min
    hi sexychat girl wait text text great night chat send stop stop servic
    hi ami send free phone number coupl day give access adult parti
    welcom select o2 servic ad benefit call special train advisor free mobil diall 402
    dear voucher holder next meal us use follow link pc 2 enjoy 2 4 1 dine experiencehttp
    urgent tri contact today draw show prize guarante call 09058094507 land line claim valid 12hr
    donat unicef asian tsunami disast support fund text donat 864233 ad next bill
    goldvik invit friend repli see stop send stop frnd 62468
    phoni award today voda number end xxxx select receiv award match pleas call 08712300220 quot claim code 3100 standard rate app
    cd 4u congratul ur award cd gift voucher gift guarante freeentri 2 wkli draw xt music 87066 tnc
    guarante cash prize claim yr prize call custom servic repres 08714712412 cost 10p
    ur current 500 pound maxim ur send go 86688 cc 08718720201
    privat 2003 account statement show 800 point call 08715203685 identifi expir
    like tell deepest darkest fantasi call 09094646631 stop text call 08712460324 nat rate
    natali invit friend repli see stop send stop frnd 62468
    jamster get free wallpap text heart 88888 c appli 16 need help call 08701213186
    free video camera phone half price line rental 12 mth 500 cross ntwk min 100 txt call mobileupd8 08001950382
    83039 uk break accommodationvouch term condit appli 2 claim mustprovid claim number 15541
    5p 4 alfi moon children need song ur mob tell ur m8 txt tone chariti 8007 nokia poli chariti poli zed 08701417012 profit 2 chariti
    win shop spree everi week start 2 play text store skilgm tscs08714740323 1winawk age16
    2nd attempt contract u week top prize either cash prize call 09066361921
    want new nokia 3510i colour phone deliveredtomorrow 300 free minut mobil 100 free text free camcord repli call 08000930705
    themob hit link get premium pink panther game new 1 sugabab crazi zebra anim badass hoodi 4 free
    msg mobil content order resent previou attempt fail due network error queri customersqueri
    1 new messag pleas call 08715205273
    decemb mobil entitl updat latest colour camera mobil free call mobil updat vco free 08002986906
    get 3 lion england tone repli lionm 4 mono lionp 4 poli 4 go 2 origin n best tone 3gbp network oper rate appli
    privat 2003 account statement 078
    4 costa del sol holiday await collect call 09050090044 toclaim sae tc pobox334 stockport sk38xh max10min
    get garden readi summer free select summer bulb seed worth scotsman saturday stop go2
    sm auction brand new nokia 7250 4 auction today auction free 2 join take part txt nokia 86021
    ree entri 2 weekli comp chanc win ipod txt pod 80182 get entri std txt rate c appli 08452810073 detail
    record indic u mayb entitl 5000 pound compens accid claim 4 free repli claim msg 2 stop txt stop
    call germani 1 penc per minut call fix line via access number 0844 861 85 prepay direct access
    mobil 11mth updat free orang latest colour camera mobil unlimit weekend call call mobil upd8 freefon 08000839402 2stoptxt
    privat 2003 account statement fone show 800 point call 08715203656 identifi code 42049 expir
    someonon know tri contact via date servic find could call mobil landlin 09064015307 box334sk38ch
    urgent pleas call 09061213237 landlin cash 4 holiday await collect cs sae po box 177 m227xi
    urgent mobil number award prize guarante call 09061790126 land line claim valid 12hr 150ppm
    urgent pleas call 09061213237 landlin cash luxuri 4 canari island holiday await collect cs sae po box m227xi 150ppm
    xma iscom ur award either cd gift voucher free entri 2 r weekli draw txt music 87066 tnc
    u r subscrib 2 textcomp 250 wkli comp 1st wk free question follow subsequ wk charg unsubscrib txt stop 2 84128 custcar 08712405020
    call 09095350301 send girl erot ecstaci stop text call 08712460324 nat rate
    import messag final contact attempt import messag wait custom claim dept expir call 08717507382
    date two start sent text talk sport radio last week connect think coincid
    current lead bid paus auction send custom care 08718726270
    free entri gr8prize wkli comp 4 chanc win latest nokia 8800 psp cash everi great 80878 08715705022
    1 new messag call
    santa call would littl one like call santa xma eve call 09058094583 book time
    guarante 32000 award mayb even cash claim ur award call free 0800 legitimat efreefon number wat u think
    latest news polic station toilet stolen cop noth go
    sparkl shop break 45 per person call 0121 2025050 visit
    txt call 86888 claim reward 3 hour talk time use phone inc 3hr 16 stop txtstop
    wml c
    urgent last weekend draw show cash spanish holiday call 09050000332 claim c rstm sw7 3ss 150ppm
    urgent tri contact last weekend draw show u prize guarante call 09064017295 claim code k52 valid 12hr 150p pm
    2p per min call germani 08448350055 bt line 2p per min check info c text stop opt
    marvel mobil play offici ultim game ur mobil right text spider 83338 game send u free 8ball wallpap
    privat 2003 account statement 07808247860 show 800 point call 08719899229 identifi code 40411 expir
    privat 2003 account statement show 800 point call 08718738001 identifi code 49557 expir
    want explicit sex 30 sec ring 02073162414 cost gsex pobox 2667 wc1n 3xx
    ask 3mobil 0870 chatlin inclu free min india cust serv sed ye l8er got mega bill 3 dont giv shit bailiff due day 3 want
    contract mobil 11 mnth latest motorola nokia etc free doubl min text orang tariff text ye callback remov record
    remind o2 get pound free call credit detail great offer pl repli 2 text valid name hous postcod
    2nd time tri 2 contact u pound prize 2 claim easi call 087187272008 now1 10p per minut
    


```python
#separating each words
spam_corpus = []
for msg in df[df['target']==1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)
```


```python
spam_corpus
```




    ['free',
     'entri',
     '2',
     'wkli',
     'comp',
     'win',
     'fa',
     'cup',
     'final',
     'tkt',
     '21st',
     'may',
     'text',
     'fa',
     '87121',
     'receiv',
     'entri',
     'question',
     'std',
     'txt',
     'rate',
     'c',
     'appli',
     '08452810075over18',
     'freemsg',
     'hey',
     'darl',
     '3',
     'week',
     'word',
     'back',
     'like',
     'fun',
     'still',
     'tb',
     'ok',
     'xxx',
     'std',
     'chg',
     'send',
     'rcv',
     'winner',
     'valu',
     'network',
     'custom',
     'select',
     'receivea',
     'prize',
     'reward',
     'claim',
     'call',
     'claim',
     'code',
     'kl341',
     'valid',
     '12',
     'hour',
     'mobil',
     '11',
     'month',
     'u',
     'r',
     'entitl',
     'updat',
     'latest',
     'colour',
     'mobil',
     'camera',
     'free',
     'call',
     'mobil',
     'updat',
     'co',
     'free',
     '08002986030',
     'six',
     'chanc',
     'win',
     'cash',
     '100',
     'pound',
     'txt',
     'csh11',
     'send',
     'cost',
     '6day',
     'tsandc',
     'appli',
     'repli',
     'hl',
     '4',
     'info',
     'urgent',
     '1',
     'week',
     'free',
     'membership',
     'prize',
     'jackpot',
     'txt',
     'word',
     'claim',
     '81010',
     'c',
     'lccltd',
     'pobox',
     '4403ldnw1a7rw18',
     'xxxmobilemovieclub',
     'use',
     'credit',
     'click',
     'wap',
     'link',
     'next',
     'txt',
     'messag',
     'click',
     'http',
     'england',
     'v',
     'macedonia',
     'dont',
     'miss',
     'news',
     'txt',
     'ur',
     'nation',
     'team',
     '87077',
     'eg',
     'england',
     '87077',
     'tri',
     'wale',
     'scotland',
     'poboxox36504w45wq',
     'thank',
     'subscript',
     'rington',
     'uk',
     'mobil',
     'charg',
     'pleas',
     'confirm',
     'repli',
     'ye',
     'repli',
     'charg',
     '07732584351',
     'rodger',
     'burn',
     'msg',
     'tri',
     'call',
     'repli',
     'sm',
     'free',
     'nokia',
     'mobil',
     'free',
     'camcord',
     'pleas',
     'call',
     '08000930705',
     'deliveri',
     'tomorrow',
     'sm',
     'ac',
     'sptv',
     'new',
     'jersey',
     'devil',
     'detroit',
     'red',
     'wing',
     'play',
     'ice',
     'hockey',
     'correct',
     'incorrect',
     'end',
     'repli',
     'end',
     'sptv',
     'congrat',
     '1',
     'year',
     'special',
     'cinema',
     'pass',
     '2',
     'call',
     '09061209465',
     'c',
     'suprman',
     'v',
     'matrix3',
     'starwars3',
     'etc',
     '4',
     'free',
     '150pm',
     'dont',
     'miss',
     'valu',
     'custom',
     'pleas',
     'advis',
     'follow',
     'recent',
     'review',
     'mob',
     'award',
     'bonu',
     'prize',
     'call',
     '09066364589',
     'urgent',
     'ur',
     'award',
     'complimentari',
     'trip',
     'eurodisinc',
     'trav',
     'aco',
     'entry41',
     'claim',
     'txt',
     'di',
     '87121',
     'morefrmmob',
     'shracomorsglsuplt',
     '10',
     'ls1',
     '3aj',
     'hear',
     'new',
     'divorc',
     'barbi',
     'come',
     'ken',
     'stuff',
     'pleas',
     'call',
     'custom',
     'servic',
     'repres',
     '0800',
     '169',
     '6031',
     'guarante',
     'cash',
     'prize',
     'free',
     'rington',
     'wait',
     'collect',
     'simpli',
     'text',
     'password',
     'mix',
     '85069',
     'verifi',
     'get',
     'usher',
     'britney',
     'fml',
     'po',
     'box',
     '5249',
     'mk17',
     '92h',
     '450ppw',
     '16',
     'gent',
     'tri',
     'contact',
     'last',
     'weekend',
     'draw',
     'show',
     'prize',
     'guarante',
     'call',
     'claim',
     'code',
     'k52',
     'valid',
     '12hr',
     '150ppm',
     'winner',
     'u',
     'special',
     'select',
     '2',
     'receiv',
     '4',
     'holiday',
     'flight',
     'inc',
     'speak',
     'live',
     'oper',
     '2',
     'claim',
     'privat',
     '2004',
     'account',
     'statement',
     '07742676969',
     'show',
     '786',
     'unredeem',
     'bonu',
     'point',
     'claim',
     'call',
     '08719180248',
     'identifi',
     'code',
     '45239',
     'expir',
     'urgent',
     'mobil',
     'award',
     'bonu',
     'caller',
     'prize',
     'final',
     'tri',
     'contact',
     'u',
     'call',
     'landlin',
     '09064019788',
     'box42wr29c',
     '150ppm',
     'today',
     'voda',
     'number',
     'end',
     '7548',
     'select',
     'receiv',
     '350',
     'award',
     'match',
     'pleas',
     'call',
     '08712300220',
     'quot',
     'claim',
     'code',
     '4041',
     'standard',
     'rate',
     'app',
     'sunshin',
     'quiz',
     'wkli',
     'q',
     'win',
     'top',
     'soni',
     'dvd',
     'player',
     'u',
     'know',
     'countri',
     'algarv',
     'txt',
     'ansr',
     '82277',
     'sp',
     'tyron',
     'want',
     '2',
     'get',
     'laid',
     'tonight',
     'want',
     'real',
     'dog',
     'locat',
     'sent',
     'direct',
     '2',
     'ur',
     'mob',
     'join',
     'uk',
     'largest',
     'dog',
     'network',
     'bt',
     'txting',
     'gravel',
     '69888',
     'nt',
     'ec2a',
     '150p',
     'rcv',
     'msg',
     'chat',
     'svc',
     'free',
     'hardcor',
     'servic',
     'text',
     'go',
     '69988',
     'u',
     'get',
     'noth',
     'u',
     'must',
     'age',
     'verifi',
     'yr',
     'network',
     'tri',
     'freemsg',
     'repli',
     'text',
     'randi',
     'sexi',
     'femal',
     'live',
     'local',
     'luv',
     'hear',
     'netcollex',
     'ltd',
     '08700621170150p',
     'per',
     'msg',
     'repli',
     'stop',
     'end',
     'custom',
     'servic',
     'annonc',
     'new',
     'year',
     'deliveri',
     'wait',
     'pleas',
     'call',
     '07046744435',
     'arrang',
     'deliveri',
     'winner',
     'u',
     'special',
     'select',
     '2',
     'receiv',
     'cash',
     '4',
     'holiday',
     'flight',
     'inc',
     'speak',
     'live',
     'oper',
     '2',
     'claim',
     '0871277810810',
     'stop',
     'bootydeli',
     'invit',
     'friend',
     'repli',
     'see',
     'stop',
     'send',
     'stop',
     'frnd',
     '62468',
     'bangbab',
     'ur',
     'order',
     'way',
     'u',
     'receiv',
     'servic',
     'msg',
     '2',
     'download',
     'ur',
     'content',
     'u',
     'goto',
     'wap',
     'bangb',
     'tv',
     'ur',
     'mobil',
     'menu',
     'urgent',
     'tri',
     'contact',
     'last',
     'weekend',
     'draw',
     'show',
     'prize',
     'guarante',
     'call',
     'claim',
     'code',
     's89',
     'valid',
     '12hr',
     'pleas',
     'call',
     'custom',
     'servic',
     'repres',
     'freephon',
     '0808',
     '145',
     '4742',
     'guarante',
     'cash',
     'prize',
     'uniqu',
     'enough',
     'find',
     '30th',
     'august',
     '500',
     'new',
     'mobil',
     '2004',
     'must',
     'go',
     'txt',
     'nokia',
     '89545',
     'collect',
     'today',
     '2optout',
     'u',
     'meet',
     'ur',
     'dream',
     'partner',
     'soon',
     'ur',
     'career',
     '2',
     'flyng',
     'start',
     '2',
     'find',
     'free',
     'txt',
     'horo',
     'follow',
     'ur',
     'star',
     'sign',
     'horo',
     'ari',
     'text',
     'meet',
     'someon',
     'sexi',
     'today',
     'u',
     'find',
     'date',
     'even',
     'flirt',
     'join',
     '4',
     '10p',
     'repli',
     'name',
     'age',
     'eg',
     'sam',
     '25',
     '18',
     'recd',
     'thirtyeight',
     'penc',
     'u',
     '447801259231',
     'secret',
     'admir',
     'look',
     '2',
     'make',
     'contact',
     'r',
     'reveal',
     'think',
     'ur',
     '09058094597',
     'congratul',
     'ur',
     'award',
     '500',
     'cd',
     'voucher',
     '125gift',
     'guarante',
     'free',
     'entri',
     '2',
     '100',
     'wkli',
     'draw',
     'txt',
     'music',
     '87066',
     'tnc',
     'tri',
     'contact',
     'repli',
     'offer',
     'video',
     'handset',
     '750',
     'anytim',
     'network',
     'min',
     'unlimit',
     'text',
     'camcord',
     'repli',
     'call',
     '08000930705',
     'hey',
     'realli',
     'horni',
     'want',
     'chat',
     'see',
     'nake',
     'text',
     'hot',
     '69698',
     'text',
     'charg',
     '150pm',
     'unsubscrib',
     'text',
     'stop',
     '69698',
     'ur',
     'rington',
     'servic',
     'chang',
     '25',
     'free',
     'credit',
     'go',
     'choos',
     'content',
     'stop',
     'txt',
     'club',
     'stop',
     '87070',
     'club4',
     'po',
     'box1146',
     'mk45',
     '2wt',
     'rington',
     'club',
     'get',
     'uk',
     'singl',
     'chart',
     'mobil',
     'week',
     'choos',
     'top',
     'qualiti',
     'rington',
     'messag',
     'free',
     'charg',
     'hmv',
     'bonu',
     'special',
     '500',
     'pound',
     'genuin',
     'hmv',
     'voucher',
     'answer',
     '4',
     'easi',
     'question',
     'play',
     'send',
     'hmv',
     '86688',
     'info',
     'custom',
     'may',
     'claim',
     'free',
     'camera',
     'phone',
     'upgrad',
     'pay',
     'go',
     'sim',
     'card',
     'loyalti',
     'call',
     '0845',
     '021',
     'end',
     'c',
     'appli',
     'sm',
     'ac',
     'blind',
     'date',
     '4u',
     'rodds1',
     'aberdeen',
     'unit',
     'kingdom',
     'check',
     'http',
     'sm',
     'blind',
     'date',
     'send',
     'hide',
     'themob',
     'check',
     'newest',
     'select',
     'content',
     'game',
     'tone',
     'gossip',
     'babe',
     'sport',
     'keep',
     'mobil',
     'fit',
     'funki',
     'text',
     'wap',
     '82468',
     'think',
     'ur',
     'smart',
     'win',
     'week',
     'weekli',
     'quiz',
     'text',
     'play',
     '85222',
     'cs',
     'winnersclub',
     'po',
     'box',
     '84',
     'm26',
     '3uz',
     'decemb',
     'mobil',
     'entitl',
     'updat',
     'latest',
     'colour',
     'camera',
     'mobil',
     'free',
     'call',
     'mobil',
     'updat',
     'co',
     'free',
     '08002986906',
     'call',
     'germani',
     '1',
     'penc',
     'per',
     'minut',
     'call',
     'fix',
     'line',
     'via',
     'access',
     'number',
     '0844',
     '861',
     '85',
     'prepay',
     'direct',
     'access',
     'valentin',
     'day',
     'special',
     'win',
     'quiz',
     'take',
     'partner',
     'trip',
     'lifetim',
     'send',
     'go',
     '83600',
     'rcvd',
     'fanci',
     'shag',
     'txt',
     'xxuk',
     'suzi',
     'txt',
     'cost',
     'per',
     'msg',
     'tnc',
     'websit',
     'x',
     'ur',
     'current',
     '500',
     'pound',
     'maxim',
     'ur',
     'send',
     'cash',
     '86688',
     'cc',
     '08708800282',
     'xma',
     'offer',
     'latest',
     'motorola',
     'sonyericsson',
     'nokia',
     'free',
     'bluetooth',
     'doubl',
     'min',
     '1000',
     'txt',
     'orang',
     'call',
     'mobileupd8',
     '08000839402',
     'discount',
     'code',
     'rp176781',
     'stop',
     'messag',
     'repli',
     'stop',
     'custom',
     'servic',
     '08717205546',
     'thank',
     'rington',
     'order',
     'refer',
     't91',
     'charg',
     'gbp',
     '4',
     'per',
     'week',
     'unsubscrib',
     'anytim',
     'call',
     'custom',
     'servic',
     '09057039994',
     'doubl',
     'min',
     'txt',
     '4',
     '6month',
     'free',
     'bluetooth',
     'orang',
     'avail',
     'soni',
     'nokia',
     'motorola',
     'phone',
     'call',
     'mobileupd8',
     '08000839402',
     '4mth',
     'half',
     'price',
     'orang',
     'line',
     'rental',
     'latest',
     'camera',
     'phone',
     '4',
     'free',
     'phone',
     '11mth',
     'call',
     'mobilesdirect',
     'free',
     '08000938767',
     'updat',
     'or2stoptxt',
     'free',
     'rington',
     'text',
     'first',
     '87131',
     'poli',
     'text',
     'get',
     '87131',
     'true',
     'tone',
     'help',
     '0845',
     '2814032',
     '16',
     '1st',
     'free',
     'tone',
     'txt',
     'stop',
     '100',
     'date',
     'servic',
     'cal',
     'l',
     '09064012103',
     'box334sk38ch',
     'free',
     'entri',
     'weekli',
     'competit',
     'text',
     'word',
     'win',
     '80086',
     '18',
     'c',
     'send',
     'logo',
     '2',
     'ur',
     'lover',
     '2',
     'name',
     'join',
     'heart',
     'txt',
     'love',
     'name1',
     'name2',
     'mobno',
     'eg',
     'love',
     'adam',
     'eve',
     '07123456789',
     '87077',
     'yahoo',
     'pobox36504w45wq',
     'txtno',
     '4',
     'ad',
     '150p',
     'someon',
     'contact',
     'date',
     'servic',
     'enter',
     'phone',
     'fanci',
     'find',
     'call',
     'landlin',
     '09111032124',
     'pobox12n146tf150p',
     'urgent',
     'mobil',
     'number',
     'award',
     'prize',
     'guarante',
     'call',
     ...]




```python
#length
len(spam_corpus)
```




    9808




```python
from collections import Counter
Counter(spam_corpus)
```




    Counter({'free': 186,
             'entri': 21,
             '2': 154,
             'wkli': 9,
             'comp': 8,
             'win': 46,
             'fa': 2,
             'cup': 3,
             'final': 13,
             'tkt': 2,
             '21st': 1,
             'may': 6,
             'text': 122,
             '87121': 2,
             'receiv': 30,
             'question': 9,
             'std': 6,
             'txt': 139,
             'rate': 26,
             'c': 43,
             'appli': 24,
             '08452810075over18': 1,
             'freemsg': 14,
             'hey': 5,
             'darl': 2,
             '3': 20,
             'week': 49,
             'word': 21,
             'back': 19,
             'like': 12,
             'fun': 8,
             'still': 5,
             'tb': 1,
             'ok': 5,
             'xxx': 10,
             'chg': 2,
             'send': 60,
             'rcv': 2,
             'winner': 13,
             'valu': 7,
             'network': 26,
             'custom': 39,
             'select': 26,
             'receivea': 1,
             'prize': 79,
             'reward': 7,
             'claim': 97,
             'call': 313,
             'code': 26,
             'kl341': 1,
             'valid': 20,
             '12': 3,
             'hour': 4,
             'mobil': 110,
             '11': 3,
             'month': 5,
             'u': 118,
             'r': 24,
             'entitl': 5,
             'updat': 12,
             'latest': 29,
             'colour': 13,
             'camera': 22,
             'co': 2,
             '08002986030': 1,
             'six': 2,
             'chanc': 22,
             'cash': 50,
             '100': 14,
             'pound': 19,
             'csh11': 1,
             'cost': 24,
             '6day': 1,
             'tsandc': 1,
             'repli': 103,
             'hl': 3,
             '4': 95,
             'info': 11,
             'urgent': 57,
             '1': 27,
             'membership': 1,
             'jackpot': 1,
             '81010': 1,
             'lccltd': 1,
             'pobox': 11,
             '4403ldnw1a7rw18': 1,
             'xxxmobilemovieclub': 1,
             'use': 12,
             'credit': 18,
             'click': 5,
             'wap': 9,
             'link': 6,
             'next': 16,
             'messag': 41,
             'http': 18,
             'england': 7,
             'v': 3,
             'macedonia': 1,
             'dont': 8,
             'miss': 7,
             'news': 8,
             'ur': 119,
             'nation': 11,
             'team': 2,
             '87077': 8,
             'eg': 10,
             'tri': 36,
             'wale': 1,
             'scotland': 1,
             'poboxox36504w45wq': 1,
             'thank': 16,
             'subscript': 5,
             'rington': 30,
             'uk': 20,
             'charg': 22,
             'pleas': 50,
             'confirm': 1,
             'ye': 12,
             '07732584351': 1,
             'rodger': 1,
             'burn': 1,
             'msg': 35,
             'sm': 22,
             'nokia': 54,
             'camcord': 15,
             '08000930705': 16,
             'deliveri': 18,
             'tomorrow': 10,
             'ac': 4,
             'sptv': 2,
             'new': 64,
             'jersey': 1,
             'devil': 1,
             'detroit': 1,
             'red': 1,
             'wing': 1,
             'play': 14,
             'ice': 1,
             'hockey': 1,
             'correct': 4,
             'incorrect': 1,
             'end': 24,
             'congrat': 5,
             'year': 8,
             'special': 15,
             'cinema': 1,
             'pass': 4,
             '09061209465': 1,
             'suprman': 1,
             'matrix3': 1,
             'starwars3': 1,
             'etc': 5,
             '150pm': 3,
             'advis': 2,
             'follow': 10,
             'recent': 4,
             'review': 3,
             'mob': 19,
             'award': 55,
             'bonu': 17,
             '09066364589': 1,
             'complimentari': 10,
             'trip': 3,
             'eurodisinc': 1,
             'trav': 1,
             'aco': 1,
             'entry41': 1,
             'di': 1,
             'morefrmmob': 1,
             'shracomorsglsuplt': 1,
             '10': 7,
             'ls1': 1,
             '3aj': 1,
             'hear': 3,
             'divorc': 1,
             'barbi': 1,
             'come': 5,
             'ken': 1,
             'stuff': 1,
             'servic': 64,
             'repres': 6,
             '0800': 9,
             '169': 1,
             '6031': 1,
             'guarante': 42,
             'wait': 16,
             'collect': 42,
             'simpli': 2,
             'password': 2,
             'mix': 2,
             '85069': 1,
             'verifi': 2,
             'get': 73,
             'usher': 1,
             'britney': 1,
             'fml': 1,
             'po': 25,
             'box': 27,
             '5249': 1,
             'mk17': 1,
             '92h': 1,
             '450ppw': 1,
             '16': 19,
             'gent': 1,
             'contact': 54,
             'last': 11,
             'weekend': 12,
             'draw': 34,
             'show': 32,
             'k52': 2,
             '12hr': 15,
             '150ppm': 28,
             'holiday': 26,
             'flight': 6,
             'inc': 7,
             'speak': 7,
             'live': 22,
             'oper': 11,
             'privat': 16,
             '2004': 7,
             'account': 18,
             'statement': 15,
             '07742676969': 1,
             '786': 3,
             'unredeem': 3,
             'point': 15,
             '08719180248': 1,
             'identifi': 15,
             '45239': 2,
             'expir': 17,
             'caller': 12,
             'landlin': 29,
             '09064019788': 1,
             'box42wr29c': 1,
             'today': 33,
             'voda': 5,
             'number': 35,
             '7548': 1,
             '350': 2,
             'match': 14,
             '08712300220': 6,
             'quot': 6,
             '4041': 1,
             'standard': 6,
             'app': 4,
             'sunshin': 5,
             'quiz': 9,
             'q': 2,
             'top': 14,
             'soni': 5,
             'dvd': 5,
             'player': 10,
             'know': 19,
             'countri': 2,
             'algarv': 1,
             'ansr': 2,
             '82277': 2,
             'sp': 6,
             'tyron': 2,
             'want': 32,
             'laid': 4,
             'tonight': 3,
             'real': 12,
             'dog': 12,
             'locat': 5,
             'sent': 10,
             'direct': 10,
             'join': 18,
             'largest': 4,
             'bt': 5,
             'txting': 7,
             'gravel': 1,
             '69888': 2,
             'nt': 1,
             'ec2a': 4,
             '150p': 25,
             'chat': 37,
             'svc': 1,
             'hardcor': 2,
             'go': 32,
             '69988': 1,
             'noth': 4,
             'must': 5,
             'age': 9,
             'yr': 14,
             'randi': 2,
             'sexi': 14,
             'femal': 1,
             'local': 5,
             'luv': 5,
             'netcollex': 4,
             'ltd': 10,
             '08700621170150p': 2,
             'per': 40,
             'stop': 108,
             'annonc': 1,
             '07046744435': 1,
             'arrang': 2,
             '0871277810810': 1,
             'bootydeli': 1,
             'invit': 9,
             'friend': 13,
             'see': 16,
             'frnd': 5,
             '62468': 6,
             'bangbab': 1,
             'order': 17,
             'way': 1,
             'download': 7,
             'content': 14,
             'goto': 5,
             'bangb': 1,
             'tv': 3,
             'menu': 1,
             's89': 1,
             'freephon': 4,
             '0808': 1,
             '145': 1,
             '4742': 1,
             'uniqu': 2,
             'enough': 2,
             'find': 21,
             '30th': 1,
             'august': 2,
             '500': 19,
             '89545': 3,
             '2optout': 4,
             'meet': 5,
             'dream': 2,
             'partner': 5,
             'soon': 2,
             'career': 1,
             'flyng': 1,
             'start': 12,
             'horo': 2,
             'star': 3,
             'sign': 2,
             'ari': 1,
             'someon': 13,
             'date': 26,
             'even': 7,
             'flirt': 5,
             '10p': 13,
             'name': 15,
             'sam': 2,
             '25': 3,
             '18': 20,
             'recd': 3,
             'thirtyeight': 1,
             'penc': 3,
             '447801259231': 1,
             'secret': 7,
             'admir': 7,
             'look': 9,
             'make': 11,
             'reveal': 12,
             'think': 11,
             '09058094597': 1,
             'congratul': 12,
             'cd': 11,
             'voucher': 28,
             '125gift': 2,
             'music': 15,
             '87066': 10,
             'tnc': 7,
             'offer': 33,
             'video': 29,
             'handset': 6,
             '750': 17,
             'anytim': 12,
             'min': 45,
             'unlimit': 10,
             'realli': 1,
             'horni': 7,
             'nake': 1,
             'hot': 13,
             '69698': 2,
             'unsubscrib': 17,
             'chang': 1,
             'choos': 9,
             'club': 19,
             '87070': 1,
             'club4': 1,
             'box1146': 1,
             'mk45': 2,
             '2wt': 2,
             'singl': 5,
             'chart': 3,
             'qualiti': 2,
             'hmv': 4,
             'genuin': 1,
             'answer': 6,
             'easi': 9,
             '86688': 13,
             'phone': 52,
             'upgrad': 4,
             'pay': 2,
             'sim': 2,
             'card': 3,
             'loyalti': 6,
             '0845': 3,
             '021': 2,
             'blind': 2,
             '4u': 3,
             'rodds1': 1,
             'aberdeen': 1,
             'unit': 1,
             'kingdom': 1,
             'check': 4,
             'hide': 1,
             'themob': 5,
             'newest': 3,
             'game': 20,
             'tone': 59,
             'gossip': 1,
             'babe': 10,
             'sport': 7,
             'keep': 5,
             'fit': 1,
             'funki': 1,
             '82468': 2,
             'smart': 2,
             'weekli': 20,
             '85222': 1,
             'cs': 34,
             'winnersclub': 1,
             '84': 1,
             'm26': 3,
             '3uz': 3,
             'decemb': 3,
             '08002986906': 2,
             'germani': 3,
             'minut': 12,
             'fix': 2,
             'line': 33,
             'via': 3,
             'access': 8,
             '0844': 2,
             '861': 2,
             '85': 2,
             'prepay': 2,
             'valentin': 3,
             'day': 26,
             'take': 15,
             'lifetim': 2,
             '83600': 3,
             'rcvd': 5,
             'fanci': 12,
             'shag': 2,
             'xxuk': 1,
             'suzi': 1,
             'websit': 1,
             'x': 8,
             'current': 11,
             'maxim': 7,
             'cc': 6,
             '08708800282': 1,
             'xma': 14,
             'motorola': 10,
             'sonyericsson': 5,
             'bluetooth': 7,
             'doubl': 13,
             '1000': 9,
             'orang': 24,
             'mobileupd8': 13,
             '08000839402': 12,
             'discount': 6,
             'rp176781': 1,
             '08717205546': 1,
             'refer': 4,
             't91': 1,
             'gbp': 1,
             '09057039994': 1,
             '6month': 1,
             'avail': 2,
             '4mth': 2,
             'half': 12,
             'price': 15,
             'rental': 11,
             '11mth': 3,
             'mobilesdirect': 2,
             '08000938767': 2,
             'or2stoptxt': 2,
             'first': 6,
             '87131': 3,
             'poli': 23,
             'true': 2,
             'help': 12,
             '2814032': 1,
             '1st': 19,
             'cal': 1,
             'l': 1,
             '09064012103': 1,
             'box334sk38ch': 2,
             'competit': 2,
             '80086': 2,
             'logo': 5,
             'lover': 2,
             'heart': 3,
             'love': 10,
             'name1': 2,
             'name2': 2,
             'mobno': 2,
             'adam': 2,
             'eve': 5,
             '07123456789': 2,
             'yahoo': 2,
             'pobox36504w45wq': 4,
             'txtno': 2,
             'ad': 4,
             'enter': 15,
             '09111032124': 1,
             'pobox12n146tf150p': 1,
             '09058094455': 1,
             'land': 16,
             '3650': 1,
             '09066382422': 1,
             'ave': 2,
             '3min': 3,
             'vari': 4,
             'close': 3,
             '300603': 1,
             'post': 4,
             'bcm4284': 1,
             'ldn': 6,
             'wc1n3xx': 3,
             'loan': 3,
             'purpos': 2,
             'homeown': 1,
             'tenant': 2,
             'welcom': 8,
             'previous': 1,
             'refus': 2,
             '1956669': 1,
             'upgrdcentr': 1,
             '0207': 2,
             '153': 2,
             '26th': 1,
             'juli': 2,
             'okmail': 1,
             'dear': 15,
             'dave': 2,
             'notic': 3,
             'tenerif': 7,
             '5000': 4,
             '09061743806': 2,
             'tc': 7,
             'sae': 17,
             'box326': 2,
             'cw25wx': 4,
             'moan': 1,
             '69888nyt': 1,
             'activ': 5,
             'term': 8,
             'condit': 4,
             'visit': 5,
             '09050002311': 1,
             'b4280703': 1,
             '40gb': 6,
             'ipod': 10,
             'mp3': 5,
             '83355': 4,
             'ibhltd': 3,
             'ldnw15h': 3,
             'boltblu': 1,
             'mono': 4,
             'poly3': 1,
             'cha': 2,
             'slide': 1,
             'yeah': 1,
             'slow': 1,
             'jamz': 1,
             'toxic': 1,
             'renew': 3,
             'pin': 1,
             'tgxxrz': 1,
             '2nd': 19,
             'attempt': 21,
             'box95qu': 3,
             'worth': 8,
             '85023': 4,
             'savamob': 8,
             'member': 6,
             '08717898035': 2,
             'sub': 5,
             'unsub': 3,
             'reciev': 1,
             'within': 4,
             '24hr': 2,
             'channel': 1,
             'teletext': 1,
             'pg': 1,
             '2003': 13,
             '07815296484': 1,
             '800': 13,
             '08718738001': 2,
             '41782': 1,
             'monthlysubscript': 1,
             'csc': 1,
             'web': 2,
             'age16': 6,
             '2stop': 1,
             'call09050000327': 2,
             'us': 7,
             'ring': 5,
             '09050005321': 1,
             '150': 8,
             'textand': 1,
             '08002988890': 1,
             'shop': 11,
             'spree': 6,
             'custcar': 7,
             '08715705022': 3,
             '2000': 2,
             '08712402050': 2,
             '10ppm': 2,
             'ag': 2,
             'promo': 2,
             '07753741225': 1,
             '08715203677': 1,
             '42478': 1,
             'import': 11,
             'announc': 5,
             '542': 3,
             '0825': 1,
             'xclusiv': 1,
             'clubsaisai': 1,
             '2morow': 1,
             'soire': 1,
             'zouk': 1,
             'nichol': 1,
             'rose': 1,
             'ladi': 3,
             '22': 1,
             'kick': 3,
             'euro2004': 3,
             'kept': 2,
             'result': 2,
             'daili': 3,
             'remov': 7,
             '83222': 2,
             'textbuddi': 2,
             'guy': 5,
             'area': 8,
             '25p': 7,
             'search': 2,
             'postcod': 3,
             'one': 10,
             '89693': 2,
             'vodafon': 4,
             '4882': 1,
             '09064019014': 1,
             'holder': 6,
             'pc': 6,
             'ts': 9,
             '80062': 5,
             '08715203694': 1,
             '40533': 2,
             'rstm': 2,
             'sw7': 2,
             '3ss': 2,
             '88800': 1,
             '89034': 1,
             'premium': 2,
             '08718711108': 1,
             'sun0819': 1,
             'hello': 4,
             'seem': 1,
             'cool': 1,
             'say': 1,
             'hi': 17,
             'gr8': 7,
             '20': 3,
             'everi': 26,
             'wk': 17,
             'opt': 10,
             '08452810071': 1,
             'sue': 2,
             'old': 2,
             'work': 3,
             'lapdanc': 1,
             'sex': 10,
             'bedroom': 1,
             'textoper': 3,
             'g2': 1,
             '1da': 1,
             '150ppmsg': 1,
             'forward': 4,
             '448712404000': 1,
             '08712404000': 1,
             'immedi': 5,
             'fantast': 7,
             'deck': 1,
             'alert': 5,
             '08714712388': 1,
             '09071512433': 2,
             'b4': 6,
             '050703': 2,
             'csbcm4235wc1n3xx': 2,
             'callcost': 2,
             'mobilesvari': 2,
             '50': 4,
             '08714712394': 1,
             'email': 2,
             'alertfrom': 1,
             'jeri': 1,
             'stewarts': 1,
             '2kbsubject': 1,
             'prescripiton': 1,
             'drvgsto': 1,
             'listen': 3,
             '123': 1,
             'nokia6650': 2,
             'txtauction': 5,
             '81151': 2,
             '4t': 3,
             'ctxt': 2,
             'subscrib': 9,
             'best': 7,
             'helplin': 2,
             '08706091795': 2,
             'realiz': 1,
             '40': 2,
             'thousand': 1,
             'run': 1,
             'around': 2,
             'tattoo': 1,
             'premier': 2,
             'romant': 1,
             'pari': 3,
             'night': 7,
             'book': 6,
             '08704439680t': 1,
             'unclaim': 1,
             '09066368327': 1,
             'claimcod': 1,
             'm39m51': 1,
             'citi': 3,
             'break': 5,
             'could': 7,
             'summer': 7,
             'store': 7,
             '88039': 2,
             'skilgm': 3,
             'tscs087147403231winawk': 2,
             '0578': 2,
             'ever': 3,
             'thought': 1,
             'good': 12,
             'life': 3,
             'perfect': 1,
             'commun': 2,
             '5': 8,
             'polyphon': 4,
             '087018728737': 1,
             'toppoli': 1,
             'tune': 1,
             'subpoli': 2,
             '81618': 1,
             'pole': 1,
             '08718727870': 2,
             '14thmarch': 1,
             'availa': 1,
             'pobox84': 4,
             'm263uz': 2,
             'no1': 5,
             '8077': 1,
             'tell': 13,
             'mate': 8,
             '36504': 4,
             'w45wq': 3,
             'cashto': 2,
             '08000407165': 2,
             'getstop': 2,
             '88222': 2,
             'php': 2,
             'rg21': 1,
             '4jx': 1,
             'either': 8,
             'gift': 16,
             'outbid': 1,
             'simonwatson5120': 1,
             'shinco': 1,
             'plyr': 1,
             'bid': 8,
             'notif': 1,
             'smsservic': 1,
             'yourinclus': 1,
             'pl': 8,
             '3qxj9': 3,
             'extra': 6,
             '9ae': 3,
             'alfi': 3,
             'moon': 3,
             'children': 3,
             'need': 11,
             'song': 3,
             'm8': 4,
             'chariti': 9,
             '8007': 16,
             'zed': 5,
             '08701417012': 3,
             'profit': 3,
             'cust': 3,
             'care': 6,
             '07821230901': 1,
             'five': 3,
             '08002888812': 2,
             'wed': 2,
             '09066350750': 1,
             'ibiza': 3,
             'await': 23,
             '434': 2,
             'sk3': 2,
             '8wp': 2,
             'ppm': 3,
             'talk': 5,
             'fall': 1,
             'world': 2,
             'discreet': 2,
             'vip': 4,
             '83110': 1,
             'suppli': 2,
             'virgin': 2,
             'record': 6,
             'mysteri': 1,
             '09061104283': 1,
             'approx': 1,
             '07808': 1,
             'xxxxxx': 1,
             '08719899217': 1,
             '41685': 2,
             'posh': 1,
             'bird': 1,
             'chap': 1,
             'user': 8,
             'trial': 1,
             'prod': 1,
             'champney': 1,
             'put': 1,
             'address': 3,
             'dob': 1,
             'asap': 6,
             'ta': 1,
             '0721072': 1,
             'till': 2,
             'drop': 1,
             '10k': 2,
             '5k': 1,
             'travel': 1,
             'ntt': 7,
             'cr01327bt': 1,
             'fixedlin': 1,
             'liverpool': 2,
             'mid': 1,
             '09058094565': 2,
             'remind': 2,
             'alreadi': 1,
             'paid': 1,
             'mymobi': 1,
             'lastest': 1,
             'stereophon': 1,
             'marley': 1,
             'dizze': 1,
             'racal': 1,
             'libertin': 1,
             'stroke': 1,
             'nookii': 1,
             'bookmark': 1,
             'januari': 1,
             'male': 3,
             'sale': 4,
             'gay': 6,
             'cheaper': 2,
             'cheap': 3,
             'peak': 2,
             '08712460324': 8,
             'money': 4,
             'lucki': 8,
             '88600': 2,
             'give': 9,
             'away': 4,
             'box403': 1,
             'w1t1ji': 1,
             'matthew': 1,
             '09063440451': 1,
             'lux': 2,
             'ppm150': 1,
             'box334': 1,
             'sk38xh': 4,
             '09061749602': 1,
             '528': 1,
             'hp20': 1,
             '1yf': 1,
             'touch': 1,
             'folk': 1,
             'compani': 3,
             'enjoy': 12,
             '08718720201': 5,
             'filthi': 2,
             'stori': 1,
             'girl': 9,
             '09050001808': 1,
             'm95': 1,
             'valid12hr': 2,
             '3g': 3,
             'videophon': 3,
             '09063458130': 2,
             'videochat': 3,
             'wid': 3,
             'java': 3,
             'dload': 3,
             'polyph': 2,
             'nolin': 3,
             'rentl': 3,
             'panason': 1,
             'bluetoothhdset': 1,
             'doublemin': 1,
             'doubletxt': 1,
             'contract': 4,
             'guess': 6,
             'somebodi': 2,
             'secretli': 2,
             'wan': 11,
             'na': 10,
             '09065394514': 1,
             'datebox1282essexcm61xn': 2,
             '09058097218': 1,
             '6': 5,
             'ls15hb': 2,
             'bloke': 1,
             'zoe': 1,
             'kickoff': 1,
             'inform': 9,
             'euro': 2,
             'eastend': 1,
             'flower': 2,
             'dot': 1,
             'compar': 1,
             'violet': 1,
             'tulip': 1,
             'lili': 1,
             'e': 1,
             'f': 3,
             '84025': 2,
             'lot': 2,
             'peopl': 3,
             'regist': 5,
             'replys150': 1,
             'ask': 5,
             'cant': 3,
             '09058091854': 1,
             'box385': 1,
             'm6': 1,
             '6wu': 1,
             '09050003091': 2,
             'c52': 2,
             'xchat': 4,
             'sipix': 3,
             'digit': 6,
             '09061221061': 1,
             '28day': 1,
             'box177': 1,
             'm221bp': 1,
             '2yr': 1,
             'warranti': 1,
             'p': 2,
             '09061790121': 2,
             '3030': 1,
             'b': 10,
             'receipt': 3,
             'an': 6,
             'elvi': 1,
             'presley': 1,
             'birthday': 1,
             'o2': 5,
             'log': 7,
             'onto': 7,
             'surpris': 5,
             '449050000301': 1,
             '09050000301': 1,
             'bore': 3,
             'speed': 1,
             'speedchat': 2,
             '80155': 1,
             'em': 1,
             'swap': 1,
             'chatter': 1,
             'chat80155': 1,
             'rcd': 1,
             '08000776320': 2,
             'part': 7,
             'survey': 1,
             'yesterday': 3,
             'howev': 1,
             'wish': 2,
             '80160': 1,
             'hmv1': 1,
             'forget': 2,
             'place': 4,
             ...})




```python
#most common 30 spam words
Counter(spam_corpus).most_common(30)
```




    [('call', 313),
     ('free', 186),
     ('2', 154),
     ('txt', 139),
     ('text', 122),
     ('ur', 119),
     ('u', 118),
     ('mobil', 110),
     ('stop', 108),
     ('repli', 103),
     ('claim', 97),
     ('4', 95),
     ('prize', 79),
     ('get', 73),
     ('new', 64),
     ('servic', 64),
     ('send', 60),
     ('tone', 59),
     ('urgent', 57),
     ('award', 55),
     ('nokia', 54),
     ('contact', 54),
     ('phone', 52),
     ('cash', 50),
     ('pleas', 50),
     ('week', 49),
     ('win', 46),
     ('min', 45),
     ('c', 43),
     ('guarante', 42)]




```python
#adding 30 common words in dataframe
pd.DataFrame(Counter(spam_corpus).most_common(30))
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>call</td>
      <td>313</td>
    </tr>
    <tr>
      <th>1</th>
      <td>free</td>
      <td>186</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>154</td>
    </tr>
    <tr>
      <th>3</th>
      <td>txt</td>
      <td>139</td>
    </tr>
    <tr>
      <th>4</th>
      <td>text</td>
      <td>122</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ur</td>
      <td>119</td>
    </tr>
    <tr>
      <th>6</th>
      <td>u</td>
      <td>118</td>
    </tr>
    <tr>
      <th>7</th>
      <td>mobil</td>
      <td>110</td>
    </tr>
    <tr>
      <th>8</th>
      <td>stop</td>
      <td>108</td>
    </tr>
    <tr>
      <th>9</th>
      <td>repli</td>
      <td>103</td>
    </tr>
    <tr>
      <th>10</th>
      <td>claim</td>
      <td>97</td>
    </tr>
    <tr>
      <th>11</th>
      <td>4</td>
      <td>95</td>
    </tr>
    <tr>
      <th>12</th>
      <td>prize</td>
      <td>79</td>
    </tr>
    <tr>
      <th>13</th>
      <td>get</td>
      <td>73</td>
    </tr>
    <tr>
      <th>14</th>
      <td>new</td>
      <td>64</td>
    </tr>
    <tr>
      <th>15</th>
      <td>servic</td>
      <td>64</td>
    </tr>
    <tr>
      <th>16</th>
      <td>send</td>
      <td>60</td>
    </tr>
    <tr>
      <th>17</th>
      <td>tone</td>
      <td>59</td>
    </tr>
    <tr>
      <th>18</th>
      <td>urgent</td>
      <td>57</td>
    </tr>
    <tr>
      <th>19</th>
      <td>award</td>
      <td>55</td>
    </tr>
    <tr>
      <th>20</th>
      <td>nokia</td>
      <td>54</td>
    </tr>
    <tr>
      <th>21</th>
      <td>contact</td>
      <td>54</td>
    </tr>
    <tr>
      <th>22</th>
      <td>phone</td>
      <td>52</td>
    </tr>
    <tr>
      <th>23</th>
      <td>cash</td>
      <td>50</td>
    </tr>
    <tr>
      <th>24</th>
      <td>pleas</td>
      <td>50</td>
    </tr>
    <tr>
      <th>25</th>
      <td>week</td>
      <td>49</td>
    </tr>
    <tr>
      <th>26</th>
      <td>win</td>
      <td>46</td>
    </tr>
    <tr>
      <th>27</th>
      <td>min</td>
      <td>45</td>
    </tr>
    <tr>
      <th>28</th>
      <td>c</td>
      <td>43</td>
    </tr>
    <tr>
      <th>29</th>
      <td>guarante</td>
      <td>42</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.barplot(x = pd.DataFrame(Counter(spam_corpus).most_common(30))[0], y = pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()

#top 30 words which are used in spam messages
```


    
![png](output_85_0.png)
    



```python
#Interpretablity is important in Machine Learning


```


```python
ham_corpus = []
for msg in df[df['target']==0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)
```


```python
ham_corpus
```




    ['go',
     'jurong',
     'point',
     'crazi',
     'avail',
     'bugi',
     'n',
     'great',
     'world',
     'la',
     'e',
     'buffet',
     'cine',
     'got',
     'amor',
     'wat',
     'ok',
     'lar',
     'joke',
     'wif',
     'u',
     'oni',
     'u',
     'dun',
     'say',
     'earli',
     'hor',
     'u',
     'c',
     'alreadi',
     'say',
     'nah',
     'think',
     'goe',
     'usf',
     'live',
     'around',
     'though',
     'even',
     'brother',
     'like',
     'speak',
     'treat',
     'like',
     'aid',
     'patent',
     'per',
     'request',
     'mell',
     'oru',
     'minnaminungint',
     'nurungu',
     'vettam',
     'set',
     'callertun',
     'caller',
     'press',
     '9',
     'copi',
     'friend',
     'callertun',
     'gon',
     'na',
     'home',
     'soon',
     'want',
     'talk',
     'stuff',
     'anymor',
     'tonight',
     'k',
     'cri',
     'enough',
     'today',
     'search',
     'right',
     'word',
     'thank',
     'breather',
     'promis',
     'wont',
     'take',
     'help',
     'grant',
     'fulfil',
     'promis',
     'wonder',
     'bless',
     'time',
     'date',
     'sunday',
     'oh',
     'k',
     'watch',
     'eh',
     'u',
     'rememb',
     '2',
     'spell',
     'name',
     'ye',
     'v',
     'naughti',
     'make',
     'v',
     'wet',
     'fine',
     'way',
     'u',
     'feel',
     'way',
     'gota',
     'b',
     'serious',
     'spell',
     'name',
     'go',
     'tri',
     '2',
     'month',
     'ha',
     'ha',
     'joke',
     'ü',
     'pay',
     'first',
     'lar',
     'da',
     'stock',
     'comin',
     'aft',
     'finish',
     'lunch',
     'go',
     'str',
     'lor',
     'ard',
     '3',
     'smth',
     'lor',
     'u',
     'finish',
     'ur',
     'lunch',
     'alreadi',
     'ffffffffff',
     'alright',
     'way',
     'meet',
     'sooner',
     'forc',
     'eat',
     'slice',
     'realli',
     'hungri',
     'tho',
     'suck',
     'mark',
     'get',
     'worri',
     'know',
     'sick',
     'turn',
     'pizza',
     'lol',
     'lol',
     'alway',
     'convinc',
     'catch',
     'bu',
     'fri',
     'egg',
     'make',
     'tea',
     'eat',
     'mom',
     'left',
     'dinner',
     'feel',
     'love',
     'back',
     'amp',
     'pack',
     'car',
     'let',
     'know',
     'room',
     'ahhh',
     'work',
     'vagu',
     'rememb',
     'feel',
     'like',
     'lol',
     'wait',
     'still',
     'clear',
     'sure',
     'sarcast',
     'x',
     'want',
     'live',
     'us',
     'yeah',
     'got',
     '2',
     'v',
     'apologet',
     'n',
     'fallen',
     'actin',
     'like',
     'spoilt',
     'child',
     'got',
     'caught',
     'till',
     '2',
     'wo',
     'go',
     'badli',
     'cheer',
     'k',
     'tell',
     'anyth',
     'fear',
     'faint',
     'housework',
     'quick',
     'cuppa',
     'yup',
     'ok',
     'go',
     'home',
     'look',
     'time',
     'msg',
     'ü',
     'xuhui',
     'go',
     'learn',
     '2nd',
     'may',
     'lesson',
     '8am',
     'oop',
     'let',
     'know',
     'roommat',
     'done',
     'see',
     'letter',
     'b',
     'car',
     'anyth',
     'lor',
     'u',
     'decid',
     'hello',
     'saturday',
     'go',
     'text',
     'see',
     'decid',
     'anyth',
     'tomo',
     'tri',
     'invit',
     'anyth',
     'pl',
     'go',
     'ahead',
     'watt',
     'want',
     'sure',
     'great',
     'weekend',
     'abiola',
     'forget',
     'tell',
     'want',
     'need',
     'crave',
     'love',
     'sweet',
     'arabian',
     'steed',
     'mmmmmm',
     'yummi',
     'see',
     'great',
     'hope',
     'like',
     'man',
     'well',
     'endow',
     'lt',
     'gt',
     'inch',
     'call',
     'messag',
     'miss',
     'call',
     'get',
     'hep',
     'b',
     'immunis',
     'nigeria',
     'fair',
     'enough',
     'anyth',
     'go',
     'yeah',
     'hope',
     'tyler',
     'ca',
     'could',
     'mayb',
     'ask',
     'around',
     'bit',
     'u',
     'know',
     'stubborn',
     'even',
     'want',
     'go',
     'hospit',
     'kept',
     'tell',
     'mark',
     'weak',
     'sucker',
     'hospit',
     'weak',
     'sucker',
     'think',
     'first',
     'time',
     'saw',
     'class',
     'gram',
     'usual',
     'run',
     'like',
     'lt',
     'gt',
     'half',
     'eighth',
     'smarter',
     'though',
     'get',
     'almost',
     'whole',
     'second',
     'gram',
     'lt',
     'gt',
     'k',
     'fyi',
     'x',
     'ride',
     'earli',
     'tomorrow',
     'morn',
     'crash',
     'place',
     'tonight',
     'wow',
     'never',
     'realiz',
     'embarass',
     'accomod',
     'thought',
     'like',
     'sinc',
     'best',
     'could',
     'alway',
     'seem',
     'happi',
     'cave',
     'sorri',
     'give',
     'sorri',
     'offer',
     'sorri',
     'room',
     'embarass',
     'know',
     'mallika',
     'sherawat',
     'yesterday',
     'find',
     'lt',
     'url',
     'gt',
     'sorri',
     'call',
     'later',
     'meet',
     'tell',
     'reach',
     'ye',
     'gauti',
     'sehwag',
     'odi',
     'seri',
     'gon',
     'na',
     'pick',
     '1',
     'burger',
     'way',
     'home',
     'ca',
     'even',
     'move',
     'pain',
     'kill',
     'ha',
     'ha',
     'ha',
     'good',
     'joke',
     'girl',
     'situat',
     'seeker',
     'part',
     'check',
     'iq',
     'sorri',
     'roommat',
     'took',
     'forev',
     'ok',
     'come',
     'ok',
     'lar',
     'doubl',
     'check',
     'wif',
     'da',
     'hair',
     'dresser',
     'alreadi',
     'said',
     'wun',
     'cut',
     'v',
     'short',
     'said',
     'cut',
     'look',
     'nice',
     'today',
     'song',
     'dedic',
     'day',
     'song',
     'u',
     'dedic',
     'send',
     'ur',
     'valuabl',
     'frnd',
     'first',
     'rpli',
     'plane',
     'give',
     'month',
     'end',
     'wah',
     'lucki',
     'man',
     'save',
     'money',
     'hee',
     'finish',
     'class',
     'hi',
     'babe',
     'im',
     'home',
     'wan',
     'na',
     'someth',
     'xx',
     'k',
     'k',
     'perform',
     'u',
     'call',
     'wait',
     'machan',
     'call',
     'free',
     'that',
     'cool',
     'gentleman',
     'treat',
     'digniti',
     'respect',
     'like',
     'peopl',
     'much',
     'shi',
     'pa',
     'oper',
     'lt',
     'gt',
     'still',
     'look',
     'job',
     'much',
     'ta',
     'earn',
     'sorri',
     'call',
     'later',
     'call',
     'ah',
     'ok',
     'way',
     'home',
     'hi',
     'hi',
     'place',
     'man',
     'yup',
     'next',
     'stop',
     'call',
     'later',
     'network',
     'urgnt',
     'sm',
     'real',
     'u',
     'get',
     'yo',
     'need',
     '2',
     'ticket',
     'one',
     'jacket',
     'done',
     'alreadi',
     'use',
     'multi',
     'ye',
     'start',
     'send',
     'request',
     'make',
     'pain',
     'came',
     'back',
     'back',
     'bed',
     'doubl',
     'coin',
     'factori',
     'got',
     'ta',
     'cash',
     'nitro',
     'realli',
     'still',
     'tonight',
     'babe',
     'ela',
     'il',
     'download',
     'come',
     'wen',
     'ur',
     'free',
     'yeah',
     'stand',
     'close',
     'catch',
     'someth',
     'sorri',
     'pain',
     'ok',
     'meet',
     'anoth',
     'night',
     'spent',
     'late',
     'afternoon',
     'casualti',
     'mean',
     'done',
     'stuff42moro',
     'includ',
     'time',
     'sheet',
     'sorri',
     'smile',
     'pleasur',
     'smile',
     'pain',
     'smile',
     'troubl',
     'pour',
     'like',
     'rain',
     'smile',
     'sum1',
     'hurt',
     'u',
     'smile',
     'becoz',
     'someon',
     'still',
     'love',
     'see',
     'u',
     'smile',
     'havent',
     'plan',
     'buy',
     'later',
     'check',
     'alreadi',
     'lido',
     'got',
     '530',
     'show',
     'e',
     'afternoon',
     'u',
     'finish',
     'work',
     'alreadi',
     'watch',
     'telugu',
     'movi',
     'wat',
     'abt',
     'u',
     'see',
     'finish',
     'load',
     'loan',
     'pay',
     'hi',
     'wk',
     'ok',
     'hol',
     'ye',
     'bit',
     'run',
     'forgot',
     'hairdress',
     'appoint',
     'four',
     'need',
     'get',
     'home',
     'n',
     'shower',
     'beforehand',
     'caus',
     'prob',
     'u',
     'see',
     'cup',
     'coffe',
     'anim',
     'pleas',
     'text',
     'anymor',
     'noth',
     'els',
     'say',
     'okay',
     'name',
     'ur',
     'price',
     'long',
     'legal',
     'wen',
     'pick',
     'u',
     'ave',
     'x',
     'am',
     'xx',
     'still',
     'look',
     'car',
     'buy',
     'gone',
     '4the',
     'drive',
     'test',
     'yet',
     'wow',
     'right',
     'mean',
     'guess',
     'gave',
     'boston',
     'men',
     'chang',
     'search',
     'locat',
     'nyc',
     'someth',
     'chang',
     'cuz',
     'signin',
     'page',
     'still',
     'say',
     'boston',
     'umma',
     'life',
     'vava',
     'umma',
     'love',
     'lot',
     'dear',
     'thank',
     'lot',
     'wish',
     'birthday',
     'thank',
     'make',
     'birthday',
     'truli',
     'memor',
     'aight',
     'hit',
     'get',
     'cash',
     'would',
     'ip',
     'address',
     'test',
     'consid',
     'comput',
     'minecraft',
     'server',
     'know',
     'grumpi',
     'old',
     'peopl',
     'mom',
     'like',
     'better',
     'lie',
     'alway',
     'one',
     'play',
     'joke',
     'dont',
     'worri',
     'guess',
     'busi',
     'plural',
     'noun',
     'research',
     'go',
     'ok',
     'wif',
     'co',
     'like',
     '2',
     'tri',
     'new',
     'thing',
     'scare',
     'u',
     'dun',
     'like',
     'mah',
     'co',
     'u',
     'said',
     'loud',
     'wa',
     'ur',
     'openin',
     'sentenc',
     'formal',
     'anyway',
     'fine',
     'juz',
     'tt',
     'eatin',
     'much',
     'n',
     'puttin',
     'weight',
     'haha',
     'anythin',
     'special',
     'happen',
     'enter',
     'cabin',
     'pa',
     'said',
     'happi',
     'boss',
     'felt',
     'special',
     'askd',
     '4',
     'lunch',
     'lunch',
     'invit',
     'apart',
     'went',
     'goodo',
     'ye',
     'must',
     'speak',
     'friday',
     'ratio',
     'tortilla',
     'need',
     'hmm',
     'uncl',
     'inform',
     'pay',
     'school',
     'directli',
     'pl',
     'buy',
     'food',
     'new',
     'address',
     'pair',
     'malarki',
     'go',
     'sao',
     'mu',
     'today',
     'done',
     '12',
     'ü',
     'predict',
     'wat',
     'time',
     'ü',
     'finish',
     'buy',
     'good',
     'stuff',
     'know',
     'yetund',
     'sent',
     'money',
     'yet',
     'sent',
     'text',
     'bother',
     'send',
     'dont',
     'involv',
     'anyth',
     'impos',
     'anyth',
     'first',
     'place',
     'apologis',
     'room',
     'hey',
     'girl',
     'r',
     'u',
     'hope',
     'u',
     'r',
     'well',
     'del',
     'r',
     'bak',
     'long',
     'time',
     'c',
     'give',
     'call',
     'sum',
     'time',
     'lucyxx',
     'k',
     'k',
     'much',
     'cost',
     'home',
     'dear',
     'call',
     'accomod',
     'first',
     'answer',
     'question',
     'haf',
     'msn',
     'yiju',
     'call',
     'meet',
     'check',
     'room',
     'befor',
     'activ',
     'got',
     'c',
     'lazi',
     'type',
     'forgot',
     'ü',
     'lect',
     'saw',
     'pouch',
     'like',
     'v',
     'nice',
     'k',
     'text',
     'way',
     'sir',
     'wait',
     'mail',
     'swt',
     'thought',
     'nver',
     'get',
     'tire',
     'littl',
     'thing',
     '4',
     'lovabl',
     'person',
     'coz',
     'somtim',
     'littl',
     'thing',
     'occupi',
     'biggest',
     'part',
     'heart',
     'gud',
     'ni8',
     'know',
     'pl',
     'open',
     'back',
     'ye',
     'see',
     'ya',
     'dot',
     'what',
     'staff',
     'name',
     'take',
     'class',
     'us',
     'call',
     'check',
     'life',
     'begin',
     'qatar',
     'pl',
     'pray',
     'hard',
     'k',
     'delet',
     'contact',
     'sindu',
     'got',
     'job',
     'birla',
     'soft',
     'wine',
     'flow',
     'never',
     'yup',
     'thk',
     'cine',
     'better',
     'co',
     'need',
     '2',
     'go',
     '2',
     'plaza',
     'mah',
     'ok',
     'ur',
     'typic',
     'repli',
     'everywher',
     'dirt',
     'floor',
     'window',
     ...]




```python
len(ham_corpus)
```




    35927




```python
from collections import Counter
Counter(ham_corpus)
```




    Counter({'go': 407,
             'jurong': 1,
             'point': 17,
             'crazi': 10,
             'avail': 13,
             'bugi': 7,
             'n': 121,
             'great': 97,
             'world': 28,
             'la': 7,
             'e': 77,
             'buffet': 2,
             'cine': 7,
             'got': 239,
             'amor': 1,
             'wat': 108,
             'ok': 218,
             'lar': 38,
             'joke': 14,
             'wif': 27,
             'u': 897,
             'oni': 4,
             'dun': 55,
             'say': 127,
             'earli': 33,
             'hor': 2,
             'c': 58,
             'alreadi': 90,
             'nah': 10,
             'think': 150,
             'goe': 26,
             'usf': 11,
             'live': 25,
             'around': 59,
             'though': 26,
             'even': 78,
             'brother': 18,
             'like': 236,
             'speak': 25,
             'treat': 19,
             'aid': 2,
             'patent': 1,
             'per': 9,
             'request': 6,
             'mell': 1,
             'oru': 2,
             'minnaminungint': 1,
             'nurungu': 1,
             'vettam': 1,
             'set': 19,
             'callertun': 4,
             'caller': 3,
             'press': 5,
             '9': 19,
             'copi': 8,
             'friend': 76,
             'gon': 59,
             'na': 96,
             'home': 152,
             'soon': 54,
             'want': 209,
             'talk': 55,
             'stuff': 44,
             'anymor': 8,
             'tonight': 58,
             'k': 107,
             'cri': 6,
             'enough': 25,
             'today': 124,
             'search': 16,
             'right': 80,
             'word': 30,
             'thank': 87,
             'breather': 1,
             'promis': 12,
             'wont': 33,
             'take': 144,
             'help': 37,
             'grant': 1,
             'fulfil': 1,
             'wonder': 37,
             'bless': 7,
             'time': 220,
             'date': 12,
             'sunday': 9,
             'oh': 111,
             'watch': 65,
             'eh': 12,
             'rememb': 35,
             '2': 288,
             'spell': 5,
             'name': 31,
             'ye': 72,
             'v': 43,
             'naughti': 6,
             'make': 129,
             'wet': 3,
             'fine': 44,
             'way': 95,
             'feel': 83,
             'gota': 1,
             'b': 56,
             'serious': 6,
             'tri': 74,
             'month': 36,
             'ha': 16,
             'ü': 173,
             'pay': 39,
             'first': 52,
             'da': 138,
             'stock': 6,
             'comin': 11,
             'aft': 19,
             'finish': 67,
             'lunch': 39,
             'str': 3,
             'lor': 159,
             'ard': 22,
             '3': 40,
             'smth': 16,
             'ur': 203,
             'ffffffffff': 1,
             'alright': 23,
             'meet': 112,
             'sooner': 4,
             'forc': 3,
             'eat': 45,
             'slice': 3,
             'realli': 83,
             'hungri': 12,
             'tho': 16,
             'suck': 7,
             'mark': 9,
             'get': 351,
             'worri': 41,
             'know': 237,
             'sick': 10,
             'turn': 12,
             'pizza': 8,
             'lol': 74,
             'alway': 53,
             'convinc': 5,
             'catch': 13,
             'bu': 27,
             'fri': 16,
             'egg': 5,
             'tea': 6,
             'mom': 21,
             'left': 30,
             'dinner': 35,
             'love': 222,
             'back': 127,
             'amp': 60,
             'pack': 4,
             'car': 43,
             'let': 86,
             'room': 35,
             'ahhh': 1,
             'work': 119,
             'vagu': 2,
             'wait': 96,
             'still': 144,
             'clear': 8,
             'sure': 76,
             'sarcast': 2,
             'x': 41,
             'us': 52,
             'yeah': 86,
             'apologet': 1,
             'fallen': 1,
             'actin': 1,
             'spoilt': 1,
             'child': 3,
             'caught': 3,
             'till': 19,
             'wo': 21,
             'badli': 1,
             'cheer': 10,
             'tell': 133,
             'anyth': 72,
             'fear': 2,
             'faint': 1,
             'housework': 1,
             'quick': 8,
             'cuppa': 1,
             'yup': 41,
             'look': 60,
             'msg': 53,
             'xuhui': 3,
             'learn': 6,
             '2nd': 7,
             'may': 40,
             'lesson': 26,
             '8am': 2,
             'oop': 10,
             'roommat': 8,
             'done': 45,
             'see': 148,
             'letter': 8,
             'decid': 25,
             'hello': 39,
             'saturday': 9,
             'text': 85,
             'tomo': 19,
             'invit': 9,
             'pl': 83,
             'ahead': 6,
             'watt': 1,
             'weekend': 27,
             'abiola': 11,
             'forget': 16,
             'need': 171,
             'crave': 12,
             'sweet': 31,
             'arabian': 1,
             'steed': 1,
             'mmmmmm': 2,
             'yummi': 3,
             'hope': 123,
             'man': 34,
             'well': 110,
             'endow': 1,
             'lt': 287,
             'gt': 288,
             'inch': 6,
             'call': 235,
             'messag': 55,
             'miss': 109,
             'hep': 1,
             'immunis': 1,
             'nigeria': 8,
             'fair': 3,
             'tyler': 6,
             'ca': 58,
             'could': 57,
             'mayb': 42,
             'ask': 121,
             'bit': 45,
             'stubborn': 1,
             'hospit': 11,
             'kept': 4,
             'weak': 7,
             'sucker': 2,
             'saw': 24,
             'class': 45,
             'gram': 4,
             'usual': 14,
             'run': 29,
             'half': 23,
             'eighth': 2,
             'smarter': 1,
             'almost': 13,
             'whole': 14,
             'second': 20,
             'fyi': 6,
             'ride': 5,
             'tomorrow': 69,
             'morn': 69,
             'crash': 4,
             'place': 53,
             'wow': 10,
             'never': 41,
             'realiz': 6,
             'embarass': 5,
             'accomod': 2,
             'thought': 45,
             'sinc': 25,
             'best': 36,
             'seem': 15,
             'happi': 106,
             'cave': 1,
             'sorri': 121,
             'give': 105,
             'offer': 9,
             'mallika': 1,
             'sherawat': 1,
             'yesterday': 20,
             'find': 50,
             'url': 3,
             'later': 100,
             'reach': 44,
             'gauti': 1,
             'sehwag': 1,
             'odi': 2,
             'seri': 4,
             'pick': 71,
             '1': 40,
             'burger': 1,
             'move': 17,
             'pain': 27,
             'kill': 6,
             'good': 215,
             'girl': 36,
             'situat': 8,
             'seeker': 1,
             'part': 18,
             'check': 46,
             'iq': 2,
             'took': 17,
             'forev': 10,
             'come': 278,
             'doubl': 5,
             'hair': 24,
             'dresser': 3,
             'said': 77,
             'wun': 7,
             'cut': 15,
             'short': 9,
             'nice': 47,
             'song': 11,
             'dedic': 2,
             'day': 195,
             'send': 120,
             'valuabl': 2,
             'frnd': 19,
             'rpli': 3,
             'plane': 3,
             'end': 37,
             'wah': 4,
             'lucki': 9,
             'save': 12,
             'money': 55,
             'hee': 13,
             'hi': 116,
             'babe': 74,
             'im': 77,
             'wan': 81,
             'someth': 69,
             'xx': 11,
             'perform': 3,
             'machan': 4,
             'free': 55,
             'that': 37,
             'cool': 41,
             'gentleman': 3,
             'digniti': 3,
             'respect': 4,
             'peopl': 39,
             'much': 109,
             'shi': 1,
             'pa': 21,
             'oper': 1,
             'job': 40,
             'ta': 17,
             'earn': 3,
             'ah': 33,
             'next': 46,
             'stop': 42,
             'network': 5,
             'urgnt': 3,
             'sm': 14,
             'real': 22,
             'yo': 36,
             'ticket': 13,
             'one': 166,
             'jacket': 2,
             'use': 59,
             'multi': 1,
             'start': 52,
             'came': 23,
             'bed': 32,
             'coin': 6,
             'factori': 1,
             'cash': 13,
             'nitro': 3,
             'ela': 2,
             'il': 10,
             'download': 11,
             'wen': 20,
             'stand': 12,
             'close': 18,
             'anoth': 33,
             'night': 110,
             'spent': 8,
             'late': 57,
             'afternoon': 26,
             'casualti': 1,
             'mean': 45,
             'stuff42moro': 1,
             'includ': 6,
             'sheet': 4,
             'smile': 55,
             'pleasur': 7,
             'troubl': 7,
             'pour': 2,
             'rain': 14,
             'sum1': 2,
             'hurt': 29,
             'becoz': 3,
             'someon': 43,
             'havent': 24,
             'plan': 53,
             'buy': 67,
             'lido': 3,
             '530': 5,
             'show': 31,
             'telugu': 2,
             'movi': 26,
             'abt': 26,
             'load': 12,
             'loan': 6,
             'wk': 9,
             'hol': 3,
             'forgot': 28,
             'hairdress': 1,
             'appoint': 5,
             'four': 2,
             'shower': 13,
             'beforehand': 1,
             'caus': 21,
             'prob': 16,
             'cup': 4,
             'coffe': 8,
             'anim': 2,
             'pleas': 75,
             'noth': 34,
             'els': 23,
             'okay': 26,
             'price': 9,
             'long': 42,
             'legal': 4,
             'ave': 5,
             'am': 1,
             'gone': 16,
             '4the': 1,
             'drive': 40,
             'test': 26,
             'yet': 48,
             'guess': 35,
             'gave': 10,
             'boston': 5,
             'men': 7,
             'chang': 31,
             'locat': 2,
             'nyc': 4,
             'cuz': 7,
             'signin': 1,
             'page': 10,
             'umma': 7,
             'life': 65,
             'vava': 3,
             'lot': 49,
             'dear': 81,
             'wish': 55,
             'birthday': 25,
             'truli': 2,
             'memor': 1,
             'aight': 33,
             'hit': 11,
             'would': 75,
             'ip': 1,
             'address': 17,
             'consid': 6,
             'comput': 9,
             'minecraft': 1,
             'server': 1,
             'grumpi': 1,
             'old': 18,
             'better': 41,
             'lie': 7,
             'play': 25,
             'dont': 123,
             'busi': 20,
             'plural': 1,
             'noun': 2,
             'research': 4,
             'co': 76,
             'new': 61,
             'thing': 109,
             'scare': 5,
             'mah': 13,
             'loud': 3,
             'wa': 3,
             'openin': 1,
             'sentenc': 2,
             'formal': 1,
             'anyway': 32,
             'juz': 25,
             'tt': 5,
             'eatin': 7,
             'puttin': 2,
             'weight': 8,
             'haha': 51,
             'anythin': 3,
             'special': 24,
             'happen': 50,
             'enter': 5,
             'cabin': 2,
             'boss': 6,
             'felt': 10,
             'askd': 6,
             '4': 162,
             'apart': 6,
             'went': 48,
             'goodo': 2,
             'must': 21,
             'friday': 14,
             'ratio': 1,
             'tortilla': 2,
             'hmm': 14,
             'uncl': 14,
             'inform': 10,
             'school': 27,
             'directli': 6,
             'food': 22,
             'pair': 1,
             'malarki': 1,
             'sao': 1,
             'mu': 15,
             '12': 4,
             'predict': 5,
             'yetund': 4,
             'sent': 51,
             'bother': 8,
             'involv': 2,
             'impos': 1,
             'apologis': 3,
             'hey': 107,
             'r': 120,
             'del': 1,
             'bak': 8,
             'sum': 2,
             'lucyxx': 1,
             'cost': 12,
             'answer': 20,
             'question': 18,
             'haf': 23,
             'msn': 2,
             'yiju': 7,
             'befor': 2,
             'activ': 3,
             'lazi': 9,
             'type': 13,
             'lect': 9,
             'pouch': 2,
             'sir': 33,
             'mail': 27,
             'swt': 5,
             'nver': 1,
             'tire': 12,
             'littl': 27,
             'lovabl': 3,
             'person': 42,
             'coz': 16,
             'somtim': 2,
             'occupi': 3,
             'biggest': 1,
             'heart': 32,
             'gud': 53,
             'ni8': 12,
             'open': 21,
             'ya': 56,
             'dot': 1,
             'what': 12,
             'staff': 2,
             'begin': 5,
             'qatar': 4,
             'pray': 9,
             'hard': 12,
             'delet': 6,
             'contact': 12,
             'sindu': 1,
             'birla': 3,
             'soft': 3,
             'wine': 11,
             'flow': 3,
             'thk': 50,
             'plaza': 2,
             'typic': 1,
             'repli': 40,
             'everywher': 2,
             'dirt': 1,
             'floor': 3,
             'window': 4,
             'shirt': 7,
             'sometim': 11,
             'mouth': 2,
             'dream': 23,
             'without': 23,
             'chore': 1,
             'joy': 8,
             'tv': 21,
             'exist': 1,
             'hail': 1,
             'mist': 1,
             'becom': 8,
             'aaooooright': 1,
             'leav': 75,
             'hous': 36,
             'interview': 5,
             'boy': 32,
             'keep': 69,
             'safe': 12,
             'envi': 1,
             'everyon': 15,
             'parent': 14,
             'hand': 18,
             'excit': 3,
             'spend': 14,
             'cultur': 1,
             'modul': 4,
             'avoid': 5,
             'missunderstd': 1,
             'wit': 15,
             'belov': 2,
             'escap': 3,
             'fanci': 6,
             'bridg': 1,
             'lager': 1,
             'complet': 16,
             'form': 2,
             'clark': 1,
             'also': 65,
             'utter': 2,
             'wast': 9,
             'axi': 1,
             'bank': 15,
             'account': 20,
             'hmmm': 11,
             'hop': 7,
             'muz': 11,
             'discuss': 7,
             'liao': 37,
             'bloodi': 4,
             'hell': 6,
             'cant': 49,
             'believ': 18,
             'surnam': 1,
             'mr': 10,
             'ill': 43,
             'clue': 1,
             'spanish': 1,
             'bath': 25,
             'carlo': 16,
             'mall': 3,
             'stay': 33,
             'til': 23,
             'smoke': 26,
             'worth': 4,
             'doesnt': 9,
             'log': 8,
             'spoke': 4,
             'maneesha': 1,
             'satisfi': 1,
             'experi': 5,
             'toll': 1,
             'lift': 7,
             'especi': 9,
             'approach': 2,
             'studi': 20,
             'gr8': 10,
             'trust': 10,
             'guy': 55,
             'handsom': 2,
             'toward': 6,
             'net': 8,
             'mummi': 4,
             'boytoy': 16,
             'awesom': 19,
             'minut': 36,
             'xma': 8,
             'radio': 3,
             'ju': 35,
             'si': 20,
             'join': 17,
             'leagu': 1,
             'touch': 17,
             'deal': 11,
             'final': 15,
             'cours': 13,
             'howev': 7,
             'suggest': 5,
             'abl': 26,
             'or': 1,
             'everi': 39,
             'stool': 1,
             'settl': 7,
             'year': 59,
             'wishin': 1,
             'mrng': 10,
             'hav': 23,
             'stori': 11,
             'hamster': 2,
             'dead': 9,
             'tmr': 28,
             '1pm': 1,
             'orchard': 11,
             'mrt': 9,
             'kate': 11,
             'babyjontet': 1,
             'txt': 13,
             'xxx': 23,
             'found': 14,
             'enc': 1,
             'buck': 7,
             'darlin': 15,
             'ive': 9,
             'colleg': 17,
             'refil': 1,
             'success': 6,
             'inr': 1,
             'decim': 20,
             'keralacircl': 1,
             'prepaid': 1,
             'balanc': 3,
             'rs': 8,
             'transact': 3,
             'id': 16,
             'kr': 1,
             'goodmorn': 14,
             'sleep': 73,
             'ga': 14,
             'alter': 1,
             '11': 4,
             'dat': 37,
             'ericsson': 2,
             'oso': 23,
             'oredi': 15,
             'straight': 4,
             'dogg': 1,
             'connect': 8,
             'refund': 1,
             'bill': 9,
             'shoot': 4,
             'big': 33,
             'readi': 35,
             'bruv': 2,
             'break': 14,
             'reward': 2,
             'semest': 10,
             'chat': 15,
             'noe': 20,
             'leh': 30,
             'sound': 26,
             'match': 5,
             'head': 22,
             'draw': 5,
             'slept': 8,
             'past': 7,
             'easi': 15,
             'sen': 4,
             'select': 4,
             'exam': 16,
             'march': 12,
             'atm': 3,
             'regist': 4,
             'os': 2,
             'ubandu': 1,
             'instal': 3,
             'disk': 1,
             'import': 14,
             'file': 7,
             'system': 7,
             'repair': 3,
             'shop': 37,
             'romant': 2,
             'nite': 21,
             'sceneri': 1,
             'appreci': 9,
             'compani': 14,
             'elama': 1,
             'po': 3,
             'mudyadhu': 1,
             'strict': 1,
             'teacher': 4,
             'bcoz': 9,
             'teach': 11,
             'conduct': 2,
             'gandhipuram': 1,
             'walk': 29,
             'cross': 2,
             'road': 11,
             'side': 13,
             'street': 8,
             'rubber': 1,
             'batteri': 7,
             'die': 17,
             'print': 5,
             'upstair': 2,
             'closer': 4,
             'wil': 18,
             'theori': 2,
             'argument': 4,
             'win': 13,
             'lose': 15,
             'argu': 3,
             'kick': 5,
             'correct': 8,
             'tomarrow': 3,
             'hear': 29,
             'laptop': 11,
             'case': 13,
             'pleassssssseeeee': 1,
             'tel': 13,
             'avent': 3,
             'sportsx': 1,
             'shine': 2,
             'meant': 14,
             'sign': 7,
             'although': 2,
             'told': 51,
             'baig': 1,
             'face': 30,
             'fr': 13,
             'thanx': 30,
             'everyth': 29,
             'commerci': 2,
             'websit': 5,
             'slipper': 3,
             'kalli': 6,
             'bat': 3,
             'inning': 3,
             'didnt': 28,
             'goodnight': 7,
             'fix': 12,
             'wake': 30,
             'dearli': 3,
             'ranjith': 2,
             'cal': 5,
             'drpd': 1,
             'deeraj': 1,
             'deepak': 1,
             '5min': 2,
             'hold': 18,
             'bcum': 2,
             'angri': 11,
             'wid': 9,
             'dnt': 7,
             'childish': 2,
             'true': 20,
             'deep': 7,
             'affect': 2,
             'care': 71,
             'luv': 29,
             'kettoda': 1,
             'manda': 1,
             'up': 2,
             '3day': 1,
             'ship': 7,
             '2wk': 1,
             'usp': 1,
             'week': 66,
             'lag': 2,
             'bribe': 1,
             'nipost': 1,
             'lem': 8,
             'necessarili': 2,
             'expect': 11,
             'headin': 2,
             'mmm': 4,
             'jolt': 2,
             'suzi': 1,
             'lover': 7,
             'park': 17,
             'mini': 2,
             'disturb': 11,
             'luton': 1,
             '0125698789': 1,
             'ring': 17,
             'h': 3,
             'dint': 6,
             'wana': 11,
             'trip': 19,
             'sometm': 1,
             'evo': 1,
             'flash': 3,
             'jealou': 3,
             'sort': 17,
             'narcot': 1,
             'sunni': 4,
             'ray': 3,
             'blue': 9,
             'bay': 2,
             'might': 34,
             'object': 1,
             'bf': 5,
             'rob': 2,
             'mack': 1,
             'gf': 2,
             'theater': 1,
             'celebr': 8,
             'full': 20,
             'swing': 12,
             'tool': 2,
             'definit': 8,
             'gdeve': 1,
             'far': 16,
             'oki': 15,
             'pass': 10,
             'last': 62,
             'ahold': 1,
             'anybodi': 5,
             'throw': 6,
             'babi': 31,
             'cruisin': 1,
             'hour': 42,
             'fone': 10,
             'jenni': 2,
             'ge': 7,
             'shall': 29,
             'updat': 6,
             'edukkukaye': 1,
             'raksha': 1,
             'sens': 7,
             'gautham': 3,
             'stupid': 9,
             'cam': 2,
             'buzi': 1,
             'accident': 4,
             'resend': 1,
             'unless': 9,
             'gurl': 1,
             'appropri': 1,
             'teas': 8,
             'plz': 18,
             'rose': 6,
             'grave': 1,
             'bslvyl': 8,
             'phone': 73,
             'somebodi': 9,
             'high': 4,
             'diesel': 1,
             'shit': 35,
             'shock': 3,
             'scari': 3,
             'imagin': 8,
             'def': 4,
             'somewher': 9,
             'taxi': 2,
             'fridg': 1,
             'meal': 4,
             'womdarful': 1,
             'actor': 2,
             'remb': 1,
             'book': 32,
             'jo': 3,
             'friendship': 15,
             'hang': 6,
             'thread': 2,
             'garag': 3,
             'key': 6,
             'bookshelf': 1,
             'accept': 5,
             'sister': 16,
             'dear1': 1,
             'best1': 1,
             'clos1': 1,
             'lvblefrnd': 1,
             'jstfrnd': 1,
             'cutefrnd': 1,
             'lifpartnr': 1,
             'belovd': 2,
             'swtheart': 1,
             'bstfrnd': 1,
             'enemi': 3,
             '2day': 4,
             'normal': 11,
             'uniqu': 2,
             'rest': 11,
             'mylif': 1,
             'wot': 24,
             'lost': 11,
             'made': 26,
             'advanc': 5,
             'pongal': 4,
             'kb': 7,
             'power': 8,
             'yoga': 7,
             'dunno': 31,
             'tahan': 2,
             'anot': 2,
             'lo': 2,
             'dude': 24,
             'afraid': 3,
             'cake': 6,
             'merri': 7,
             'christma': 12,
             'kiss': 36,
             'cud': 5,
             'ppl': 5,
             'gona': 3,
             'l8': 1,
             'buse': 2,
             'waitin': 4,
             'pete': 10,
             'guild': 1,
             'bristol': 2,
             'flight': 2,
             'problem': 40,
             'track': 6,
             'record': 4,
             'read': 21,
             'women': 2,
             'light': 12,
             'apo': 2,
             'return': 12,
             'immedi': 3,
             'chanc': 11,
             'evapor': 1,
             'violat': 2,
             ...})




```python
#most common 30 ham words
Counter(ham_corpus).most_common(30)
```




    [('u', 897),
     ('go', 407),
     ('get', 351),
     ('2', 288),
     ('gt', 288),
     ('lt', 287),
     ('come', 278),
     ('got', 239),
     ('know', 237),
     ('like', 236),
     ('call', 235),
     ('love', 222),
     ('time', 220),
     ('ok', 218),
     ('good', 215),
     ('want', 209),
     ('ur', 203),
     ('day', 195),
     ('ü', 173),
     ('need', 171),
     ('one', 166),
     ('4', 162),
     ('lor', 159),
     ('home', 152),
     ('think', 150),
     ('see', 148),
     ('take', 144),
     ('still', 144),
     ('da', 138),
     ('tell', 133)]




```python
sns.barplot(x = pd.DataFrame(Counter(ham_corpus).most_common(30))[0], y = pd.DataFrame(Counter(ham_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()
```


    
![png](output_92_0.png)
    



```python
# Text Vectorization
# Using Bag of Words
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>target</th>
      <th>text</th>
      <th>num_characters</th>
      <th>num_words</th>
      <th>num_sentences</th>
      <th>transformed_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
      <td>111</td>
      <td>24</td>
      <td>2</td>
      <td>go jurong point crazi avail bugi n great world...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>Ok lar... Joking wif u oni...</td>
      <td>29</td>
      <td>8</td>
      <td>2</td>
      <td>ok lar joke wif u oni</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
      <td>155</td>
      <td>37</td>
      <td>2</td>
      <td>free entri 2 wkli comp win fa cup final tkt 21...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>U dun say so early hor... U c already then say...</td>
      <td>49</td>
      <td>13</td>
      <td>1</td>
      <td>u dun say earli hor u c alreadi say</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
      <td>61</td>
      <td>15</td>
      <td>1</td>
      <td>nah think goe usf live around though</td>
    </tr>
  </tbody>
</table>
</div>



# 4. Model Building

### We need to covert the text into numbers or Vectors. (Vectorize)

### 1. Bag of words
### 2. TF-IDF or Term Frequency–Inverse Document Frequency, 
### 3.  Word2Vec


```python
#Bag of Words
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
cv = CountVectorizer()

tfidf = TfidfVectorizer()
```


```python
#X = cv.fit_transform(df['transformed_text']).toarray()

X = tfidf.fit_transform(df['transformed_text']).toarray()
```


```python
X
```




    array([[0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           ...,
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.]])




```python
X.shape

#(SMS , words)
```




    (5160, 6782)




```python
y = df['target'].values
```


```python
y
```




    array([0, 0, 1, ..., 0, 0, 0])




```python
from sklearn.model_selection import train_test_split
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=2)
```


```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
```


```python
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()
```


```python
#GaussianNB
gnb.fit(X_train, y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test, y_pred1))
print(confusion_matrix(y_test, y_pred1))
print(precision_score(y_test, y_pred1))  #precision score should be more
```

    0.874031007751938
    [[804 112]
     [ 18  98]]
    0.4666666666666667
    


```python
#MultinomialNB
mnb.fit(X_train, y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test, y_pred2))
print(confusion_matrix(y_test, y_pred2))
print(precision_score(y_test, y_pred2))
```

    0.9641472868217055
    [[916   0]
     [ 37  79]]
    1.0
    


```python
#BernoulliNB
bnb.fit(X_train, y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test, y_pred3))
print(confusion_matrix(y_test, y_pred3))
print(precision_score(y_test, y_pred3))
```

    0.9748062015503876
    [[913   3]
     [ 23  93]]
    0.96875
    


```python
# Since precision score is max in mnb, so we will go with mnb
# tfidf --> MNB
```


```python
pip install xgboost

```

    Requirement already satisfied: xgboost in c:\users\happy\anaconda3\lib\site-packages (1.7.6)
    Requirement already satisfied: numpy in c:\users\happy\anaconda3\lib\site-packages (from xgboost) (1.24.3)
    Requirement already satisfied: scipy in c:\users\happy\anaconda3\lib\site-packages (from xgboost) (1.10.1)
    Note: you may need to restart the kernel to use updated packages.
    


```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
```


```python
svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50,random_state=2)
xgb = XGBClassifier(n_estimators=50,random_state=2)
```


```python
#made dictionary
clfs = {
    'SVC' : svc,
    'KN' : knc, 
    'NB': mnb, 
    'DT': dtc, 
    'LR': lrc, 
    'RF': rfc, 
    'AdaBoost': abc, 
    'BgC': bc, 
    'ETC': etc,
    'GBDT':gbdt,
    'xgb':xgb
}
```


```python
def train_classifier(clf,X_train,y_train,X_test,y_test):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    
    return accuracy,precision
```


```python
train_classifier(svc,X_train,y_train,X_test,y_test)

#96% accuracy
```




    (0.9689922480620154, 0.9285714285714286)




```python
df.head()
```


```python
accuracy_scores = []
precision_scores = []

for name,clf in clfs.items():
    
    current_accuracy,current_precision = train_classifier(clf, X_train,y_train,X_test,y_test)
    
    print("For ",name)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)
```


```python
performance_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision'
```


```python
performance_df
```


```python
performance_df1 = pd.melt(performance_df, id_vars = "Algorithm")
```


```python
performance_df1
```


```python
sns.catplot(x = 'Algorithm', y='value', 
               hue = 'variable',data=performance_df1, kind='bar',height=5)
plt.ylim(0.5,1.0)
plt.xticks(rotation='vertical')
plt.show()
```


```python
# model improve
# 1. Change the max_features parameter of TfIdf
```


```python
temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_max_ft_3000':accuracy_scores,'Precision_max_ft_3000':precision_scores}).sort_values('Precision_max_ft_3000',ascending=False)

```


```python
temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_scaling':accuracy_scores,'Precision_scaling':precision_scores}).sort_values('Precision_scaling',ascending=False)

```


```python
new_df = performance_df.merge(temp_df,on='Algorithm')
```


```python
new_df_scaled = new_df.merge(temp_df,on='Algorithm')
```


```python
temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_num_chars':accuracy_scores,'Precision_num_chars':precision_scores}).sort_values('Precision_num_chars',ascending=False)

```


```python
new_df_scaled.merge(temp_df,on='Algorithm')
```


```python
# Voting Classifier
svc = SVC(kernel='sigmoid', gamma=1.0,probability=True)
mnb = MultinomialNB()
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)

from sklearn.ensemble import VotingClassifier
```


```python
voting = VotingClassifier(estimators=[('svm', svc), ('nb', mnb), ('et', etc)],voting='soft')
```


```python
voting.fit(X_train,y_train)
```


```python
y_pred = voting.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))
```


```python
# Applying stacking
estimators=[('svm', svc), ('nb', mnb), ('et', etc)]
final_estimator=RandomForestClassifier()
```


```python
from sklearn.ensemble import StackingClassifier
```


```python
clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)
```


```python
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))

```


```python
import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))
```


```python

```
