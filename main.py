#to creature gui, we will use Tkinter
import tkinter as tk
#add cooler/mor emodern widgets with ttk
from tkinter import ttk
#importing datasets via pandas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#using a KNN model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from tkinter import PhotoImage
#from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer #using to encode multiple labels in a single data cell
import ast #used for parsing data strings
#import threading
import numpy as np

#//////////////////////////////////loading and preparing data/////////////////////
#load dataframe with movie dataset
df = pd.read_csv("tmdb_5000_movies.csv")

#dropping bad rows
df = df[df["budget"] > 0]
df = df[df["revenue"] > 0]

def extract_names(text):
    if pd.isna(text): #handles NaN
        return None
    
    try:
        data = ast.literal_eval(text) #convert string → list of dicts ("dictionaries")

        if not data: #handles [] -> if 'data' is an empty list
            return None

        return [item["name"] for item in data] #"item" represents a value that is branched within '"name"'.
        #in other words, we only want to extract '"name" values' -> so related datacells can be read as cleaned arrays instead for our model
    except:
        
        return None  

#cleaning data columns
df["genres"] = df["genres"].apply(extract_names)
df["genres"] = df["genres"].apply(lambda x: x if x is not None else ["Unknown"])
df["keywords"] = df["keywords"].apply(extract_names)
df["keywords"] = df["keywords"].apply(lambda x: x if x is not None else ["Unknown"])
df["production_countries"] = df["production_countries"].apply(extract_names)
df["production_countries"] = df["production_countries"].apply(lambda x: x if x is not None else ["Unknown"])
#df["revenue_clean"] = df["revenue"].replace(0, df["revenue"].median())
df["vote_average"] = df["vote_average"].replace(0, df["vote_average"].median())
#df["popularity"] = df["popularity"].replace(0, df["popularity"].median())
#df["budget_clean"] = df["budget"].replace(0, df["budget"].median())
df["runtime"] = df["runtime"].replace(0, df["runtime"].median())

mlb_genres = MultiLabelBinarizer()
genres_encoded = mlb_genres.fit_transform(df["genres"])
genres_df = pd.DataFrame(genres_encoded, columns=mlb_genres.classes_, index=df.index)
df = pd.concat([df, genres_df], axis=1)

mlb_keywords = MultiLabelBinarizer()
keywords_encoded = mlb_keywords.fit_transform(df["keywords"])
keywords_df = pd.DataFrame(keywords_encoded, columns=mlb_keywords.classes_, index=df.index)
df = pd.concat([df, keywords_df], axis=1)

mlb_country = MultiLabelBinarizer()
country_encoded = mlb_country.fit_transform(df["production_countries"])
country_df = pd.DataFrame(country_encoded, columns=mlb_country.classes_, index=df.index)
df = pd.concat([df, country_df], axis=1)

threshold = 2.5
df["success"] = (df["revenue"] >= threshold * df["budget"]).astype(int) #succes of a film is determined by if
#the film's revenue was 2.5 times or greater than the film's initial budget or not
#a given film's succes is determined if the given contition for the budget multiplied by 2.5 will likely
#generate the acceptable revenue )because if we include revenue this will cause "data leakage" (basically
#the model will be cheating)

#choosing target features
features = list(genres_df.columns) + \
           list(keywords_df.columns) + \
           list(country_df.columns) + \
           ["vote_average", "budget", "runtime"]
target = 'success'

#creating feature matrix
X = df[features]
y = df[target]
print(df["success"].value_counts(normalize=True))
features = X.columns.tolist()
#////////////////////////////////model preparations////////////////////////
#split model into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

knn_model = KNeighborsClassifier(n_neighbors=5) #we are using 5 neighbors (which is a common default value for KNN)
#the 5 closest passengers in the training data to the passenger we want to predict for

knn_model.fit(X_train, y_train)
knn_accuracy = accuracy_score(y_test, knn_model.predict(X_test))

y_pred = knn_model.predict(X_test)


#////////////////////////////////////////predictions////////////////////////
def predict(): 
    #Create empty row with ALL features
    row = pd.DataFrame(0, index=[0], columns=features)

    #Numeric inputs
    row["budget"] = int(budget_entry.get())
    row["runtime"] = int(runtime_entry.get())
    row["vote_average"] = float(rating_entry.get())
#if "g"/"k"/"c" is set to 1, then that means the user entered it in and is active, else 0 - not entered
    #Genres
    selected_genres = [genres_listbox.get(i) for i in genres_listbox.curselection()]
    for g in selected_genres:
        if g in row.columns:
             row.at[0, g] = 1
    print("Selected genres:", selected_genres)
    print("Matching columns:", [g for g in selected_genres if g in row.columns])
    #Keywords
    selected_keywords = [keywords_listbox.get(i) for i in keywords_listbox.curselection()]
    for k in selected_keywords:
        if k in row.columns:
             row.at[0, k] = 1
    print("Selected keywords:", selected_keywords)
    print("Matching columns:", [g for g in selected_keywords if g in row.columns])
    #Countries
    selected_countries = [country_listbox.get(i) for i in country_listbox.curselection()]
    for c in selected_countries:
        if c in row.columns:
             row.at[0, c] = 1

    knn_result = knn_model.predict(row)[0]
    
    if (knn_result == 1): 
        knn_result_label.config(text="KNN: Success", fg="green",font=("Arial",20,"bold")) #"fg" stands for foreground
    else:
        knn_result_label.config(text="KNN: Flopped", fg="red",font=("Arial",20,"bold")) #creating custom config output

#//////////////////////////////////////////building GUI/////////////////////////////////////
print("Reached GUI section")
#creating main application window
root = tk.Tk()
#set window title
root.title("Movie Success Predictor using ML")
#creating window dimensions
root.geometry("1300x800")
root.config(bg="blue") #setting bacground color

#creating label for title
tk.Label(root, text="Movie Success Predictor", bg="purple", fg="yellow", font=("Arial",24,"bold")).pack(pady=20)
#creating accuracy label
tk.Label(root, text=f"KNN Accuracy {knn_accuracy:.1%}", bg="white", fg="green", font=("Arial",14,"bold")).pack(pady=10)

#////////////////////////////////////creating input fields////////////////////////
#creating frame for holding input fields
frame = tk.Frame(root, bg="purple")
frame.pack(pady=15) #adding frame to window

#creating function that creates labled input fields
def addfield(lable_text, variable, options=None, default=""):
    #create frame to hole input & label
    row = tk.Frame(frame, bg="blue")
    row.pack(fill="x", pady=4) #adding to window
    #creating lable and placing on window
    tk.Label(row, text=lable_text, width=44, anchor='w', bg="purple", fg="yellow", font=("Arial",12)).pack(side=tk.LEFT)
    #"anchor" is directional (north, west, south, southeast, etc)
    if options:
        ttk.Combobox(row, textvariable=variable, values=options, state="readonly", width=15).pack(side=tk.LEFT)
    else: 
        tk.Entry(row, textvariable=variable, width=18).pack(side=tk.LEFT)

#creating the tk variables to hold the input values
#below is just creating some default values
budget_entry = tk.StringVar(value="10000000")  
runtime_entry = tk.StringVar(value="10") 
rating_entry = tk.StringVar(value="5") 
#popularity_entry = tk.StringVar(value="C") 
genres_entry = tk.StringVar(value="Unknown") 
keywords_entry = tk.StringVar(value="Unknown") 
country_entry = tk.StringVar(value="Unknown") 

#using addfield function to create input fields for user
addfield("Budget of Movie", budget_entry)
addfield("Runtime of Movie", runtime_entry)
addfield("1.0-10.0 Average Rating of Movie", rating_entry) 
#addfield("Movie Genres (Separate with commas)", genres_entry)
#addfield("Movie Keywords (Separate with commas)", keywords_entry)
#addfield("Movie Countries Produced In (Separate with commas and use official names)", country_entry)

tk.Label(frame, text="Genres", bg="purple", fg="yellow").pack()
genres_listbox = tk.Listbox(frame, selectmode=tk.MULTIPLE, height=6, exportselection=False)
genres_listbox.pack()
for genre in mlb_genres.classes_:
    genres_listbox.insert(tk.END, genre)

tk.Label(frame, text="Keywords", bg="purple", fg="yellow").pack()
keywords_listbox = tk.Listbox(frame, selectmode=tk.MULTIPLE, height=6, exportselection=False)
keywords_listbox.pack()
for k in mlb_keywords.classes_:
    keywords_listbox.insert(tk.END, k)

tk.Label(frame, text="Countries", bg="purple", fg="yellow").pack()
country_listbox = tk.Listbox(frame, selectmode=tk.MULTIPLE, height=6, exportselection=False)
country_listbox.pack()
for c in mlb_country.classes_:
    country_listbox.insert(tk.END, c)    

#//////////////////////////////////creating button to trigger predictions///////////////////
tk.Button(root, text="Predict Success", command=predict, font=("Arial", 14, "bold"), fg="purple", bg="yellow",
          activebackground="lightgreen", activeforeground="black").pack(pady=15)
#creating label to display prediction result
knn_result_label = tk.Label(root, text="Enter a Movie's profile and click predict", bg="white", fg="green",
                        font=("Arial",14))
knn_result_label.pack(pady=10)
#the above label will be updated when the predict button is pressed/predict function is called

#Permutation importance is a model-agnostic technique used to measure the contribution of each feature to a fitted model's performance
#knn_perm_imp = permutation_importance(knn_model, X_test, y_test, n_repeats=10, random_state=42)
#knn_imp = sorted(zip(features, knn_perm_imp.importances_mean), key=lambda x: -x[1])
#knn_imp_text = "  |  ".join(f"{n}: {s:.0%}" for n, s in knn_imp)
#tk.Label(root, text=f"KNN: Top Factors For Survival: {knn_imp_text}", bg="purple", fg="white", font=("Arial", 10)).pack(pady=10)





#def calculate_importance():
#    knn_perm_imp = permutation_importance(knn_model, X_test, y_test, n_repeats=5, random_state=42)
#    knn_imp = sorted(zip(features, knn_perm_imp.importances_mean), key=lambda x: -x[1])
#    knn_imp_text = "  |  ".join(f"{n}: {s:.0%}" for n, s in knn_imp)

    # Schedule UI update safely on main thread
#    root.after(0, lambda: tk.Label(
#        root,
#        text=f"KNN: Top Factors: {knn_imp_text}", bg="purple", fg="white", font=("Arial", 10)
#    ).pack(pady=10))

#threading.Thread(target=calculate_importance, daemon=True).start() #running seperate from main thread
#so loading doesn't take a million years


#the below displays "pseudo-weights." It displays which features are most commonly “on” in successful samples
#This is the best solution I could find that could give the user some sort of look into how certain featurers
#are werighted without absolutely tanking performance
feature_importance = X.mean().sort_values(ascending=False)
top_features = feature_importance.head(10)

importance_text = " | ".join(f"{f}: {v:.2f}" for f, v in top_features.items())

tk.Label(root, text=f"Top Active Features: {importance_text}",
         bg="purple", fg="white", font=("Arial", 10)).pack(pady=10)
#////////////////////////////////////////////running GUI and main loop////////////////////////////////////
#app starts
root.mainloop()

