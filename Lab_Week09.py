# Week 9 Lab
# Azarm Piran | A01195657


# Example 1: Installing Speech Recognition


# Example 2: Speech-to-Text

import speech_recognition as sr
PATH = "C:\\Users\\Azarm\\Desktop\\BCIT\\AdvancedTopics\\DataSets\\audiosamples1/elaine_0.wav"

r = sr.Recognizer()

audioFile = sr.AudioFile(PATH)
with audioFile as source:
    audio  = r.record(source)
    output = r.recognize_google(audio)
    print(output)



# Duration


# Example 3: Breaking the audio clip into sections with duration

import speech_recognition as sr
PATH = "C:\\Users\\Azarm\\Desktop\\BCIT\\AdvancedTopics\\DataSets\\/audiosamples1/kramer_3.wav"

r = sr.Recognizer()

jackhammer = sr.AudioFile(PATH)
with jackhammer as source:
    audio1 = r.record(source, duration=1.5)
    audio2 = r.record(source, duration=1.2)

    # Shows first section.
    output1 = r.recognize_google(audio1)
    print(output1)

    # Shows second section.
    output2 = r.recognize_google(audio2)
    print(output2)




# Offset

# Example 4: Offset

import speech_recognition as sr
PATH = "C:\\Users\\Azarm\\Desktop\\BCIT\\AdvancedTopics\\DataSets\\/audiosamples1/kramer_3.wav"

r = sr.Recognizer()

jackhammer = sr.AudioFile(PATH)
with jackhammer as source:
    audio = r.record(source, offset=1.5, duration=1.5)
    output = r.recognize_google(audio)
    print(output)



# Using the Microphone


# Example 5: Testing the Microphone

import speech_recognition as sr
r   = sr.Recognizer()
mic = sr.Microphone()

with mic as source:
    # Add the following line to filter out background noise.
    # r.adjust_for_ambient_noise(source)
    audio = r.listen(source)

output = r.recognize_google(audio)
print(output)



# Exercise 1









# Cluster Analysis


# K-means Clustering


# Example 6: Calculating the Centroid


# Example 7: K-Means Clustering Introduction Code

from    pandas import DataFrame
import  matplotlib.pyplot as plt
from    sklearn.cluster import KMeans

Data = {
          # 2 2 2 2 2 2 2 2 2 2
    'x': [25, 34, 22, 27, 33, 33, 31, 22, 35, 34, \
          # 0 0 0 0 0 0 0 0 0 0
          67, 54, 57, 43, 50, 57, 59, 52, 65, 47, \
          # 1 1 1 1 1 1 1 1 1 1
          49, 48, 35, 33, 44, 45, 38, 43, 51, 46],

          # 2 2 2 2 2 2 2 2 2 2
    'y': [79, 51, 53, 78, 59, 74, 73, 57, 69, 75, \
          # 0 0 0 0 0 0 0 0 0 0
          51, 32, 40, 47, 53, 36, 35, 58, 59, 50, \
          # 1 1 1 1 1 1 1 1 1 1
          25, 20, 14, 12, 20, 5, 29, 27, 8, 7]}

# Perform clustering.
df         = DataFrame(Data, columns=['x', 'y'])
kmeans     = KMeans(n_clusters=3,  random_state=142).fit(df)
centroids  = kmeans.cluster_centers_

# Show centroids.
print("\n*** centroids ***")
print(centroids)

# Show sample labels.
print("\n*** sample labels ***")
print(kmeans.labels_)

# Parameters: [c for color, s for dot size]
plt.scatter(df['x'], df['y'], c=kmeans.labels_, s=50, alpha=0.5)

# Shows the 3 centroids in red.
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, alpha=0.3)
plt.show()





# Exercise 3








# Exercise 6

from    pandas import DataFrame
import  matplotlib.pyplot as plt
from    sklearn.cluster import KMeans

Data = {

    'x': [25, 34, 22, 27, 33, 33, 31, 22, 35, 34, 67, 54, 57, 43, 50, 57, 59, 52, 65, 47,49, 48, 35, 33, 44, 45, 38, 43, 51, 46],
    'y': [79, 51, 53, 78, 59, 74, 73, 57, 69, 75, 51, 32, 40, 47, 53, 36, 35, 58, 59, 50, 25, 20, 14, 12, 20, 5, 29, 27, 8, 7]}

# Perform clustering.
df         = DataFrame(Data, columns=['x', 'y'])
kmeans     = KMeans(n_clusters=4,  random_state=142).fit(df)
centroids  = kmeans.cluster_centers_

# Show centroids.
print("\n*** centroids ***")
print(centroids)

# Show sample labels.
print("\n*** sample labels ***")
print(kmeans.labels_)

# Parameters: [c for color, s for dot size]
plt.scatter(df['x'], df['y'], c=kmeans.labels_, s=50, alpha=0.5)

# Shows the 3 centroids in red.
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, alpha=0.3)
plt.show()





# Example 8: Data Scoring with K-Means

# Example 8: Data Scoring with K-Means