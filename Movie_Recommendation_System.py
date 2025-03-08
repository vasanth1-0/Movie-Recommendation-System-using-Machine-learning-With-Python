
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#loading the date from csv file to append data frames
movies_data = pd.read_csv('C:\\Users\\DELL\\OneDrive\\Documents\\Vasanth\\Projects\\Python\\Machine Learning\\Movie Recommendation System\\movies.csv')

#Select the relevant features for recommendation
selected_features=['genres','keywords','cast','director','tagline','release_date']

#combine all 6 selected features
combined_features = movies_data['genres']+' '+ movies_data['keywords']+' '+movies_data['cast']+' '+movies_data['director']+' '+movies_data['tagline']+' '+movies_data['release_date']

combined_features = combined_features.fillna('') 

#Converting the text data to feature vectors
vectorizer=TfidfVectorizer()

feature_vectors=vectorizer.fit_transform(combined_features)

#using Cosine alogo
similarity=cosine_similarity(feature_vectors)

#Getting input from the user
movie_name=input('Enter Movie Name:')

list_of_all_movie_titles=movies_data['title'].tolist()

#find closest match to which user has enterd
find_close_match=difflib.get_close_matches(movie_name,list_of_all_movie_titles)

if find_close_match:  # Checks if the list is not empty
    close_match = find_close_match[0]
    print(close_match)
else:  # Executes if the list is empty
    print("Not found")

#find index of movie based on title
index_of_movie=movies_data[movies_data.title==close_match]['index'].values[0]

#list of similar movies score
similarity_score=list(enumerate(similarity[index_of_movie]))

#sort movies based on similarity score
sorted_similar_movies=sorted(similarity_score,key=lambda x:x[1],reverse=True)

#print name of top 10 similar movies based on the index
print('Movies Recommended based on your search:\n')

i=1
for movie in sorted_similar_movies:
  index=movie[0]
  title_from_index=movies_data[movies_data.index==index]['title'].values[0]
  if(i<=10):
    print(i,title_from_index)
    i+=1
