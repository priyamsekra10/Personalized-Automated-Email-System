import json
import re
import nltk
import warnings
import numpy as np
import pandas as pd
import requests
import random
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, label_binarize, StandardScaler
from collections import Counter
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score
from pymongo import MongoClient
warnings.filterwarnings("ignore")


user_df = pd.read_csv('user_dataset_20.csv')


new_user = user_df.drop('emp_id ', axis = 1)
new_user = new_user.drop('name', axis = 1)
new_user = new_user.drop('team', axis = 1)
new_user = new_user.drop('company', axis = 1)
new_user = new_user.drop('cyber_score', axis = 1)
new_user = new_user.drop('error_rate', axis = 1)


new_user["interests"] = new_user["interests"].apply(lambda x: [item.strip() for item in x.split(",")])
new_user["template_id_fished"] = new_user["template_id_fished"].apply(lambda x: [item.strip() for item in x.split(",")])

new_user = pd.DataFrame({
    "Language": new_user["Language"].repeat(new_user["interests"].str.len()),
    "interests": [item for sublist in new_user["interests"] for item in sublist],
    "template_id_fished": new_user["template_id_fished"].repeat(new_user["interests"].str.len()),
})

new_user = pd.DataFrame({
    "template_id_fished": [item for sublist in new_user["template_id_fished"] for item in sublist],
    "Language": new_user["Language"].repeat(new_user["template_id_fished"].str.len()),
    "interests": new_user["interests"].repeat(new_user["template_id_fished"].str.len())
    
})

api = 'https://eobkthtzq6.execute-api.eu-west-3.amazonaws.com/dev/api/phishingtemplate'

response = requests.get(api)
data = response.json()
temp_df = pd.DataFrame(data['data'])



temp_df["tempdetails"] = temp_df["tempdetails"].apply(lambda x: x.split(", "))

temp_df = pd.DataFrame({
    "_id": temp_df["_id"].repeat(temp_df["tempdetails"].str.len()),
    "tempdetails": [item for sublist in temp_df["tempdetails"] for item in sublist],
    "message": temp_df["message"].repeat(temp_df["tempdetails"].str.len()),
    "companyName": temp_df["companyName"].repeat(temp_df["tempdetails"].str.len()),
    "imagePath": temp_df["imagePath"].repeat(temp_df["tempdetails"].str.len()),
    "templateName": temp_df["templateName"].repeat(temp_df["tempdetails"].str.len()),
    "templatePath": temp_df["templatePath"].repeat(temp_df["tempdetails"].str.len()),
    "subject": temp_df["subject"].repeat(temp_df["tempdetails"].str.len()),
    "attachmentname": temp_df["attachmentname"].repeat(temp_df["tempdetails"].str.len()),
    "attachmentpath": temp_df["attachmentpath"].repeat(temp_df["tempdetails"].str.len()),
    "domain": temp_df["domain"].repeat(temp_df["tempdetails"].str.len()),
    "difficultylevel": temp_df["difficultylevel"].repeat(temp_df["tempdetails"].str.len()), 
    "__v": temp_df["__v"].repeat(temp_df["tempdetails"].str.len())     
})
# # ------------------------------------------------------------------------------------------------


joined_df = new_user[['interests','template_id_fished']]



label_encoder = LabelEncoder()
interests = joined_df['interests']
label_encoder.fit(interests)
encoded_interests = label_encoder.transform(interests)
label_interest_mapping = dict(zip(encoded_interests, joined_df['interests']))
joined_df['interests'] = encoded_interests



label_encoder = LabelEncoder()

# Assuming you have the interests data in a list or pandas Series called "interests"
temp = joined_df['template_id_fished']

# Fit the LabelEncoder on the interests data
label_encoder.fit(temp)

# Transform the interests data using label encoding
encoded_temp = label_encoder.transform(temp)
label_interest_mapping1 = dict(zip(encoded_temp, joined_df['template_id_fished']))

# Print the label-interest mapping
# for label, interest in label_interest_mapping.items():
#      print(f"Label {label}: {interest}")
joined_df['template_id_fished'] = encoded_temp

df = joined_df


# Split the data into X and y
X = df[['interests']]
y = df['template_id_fished']

# Create and train the RandomForestClassifier model
model = RandomForestClassifier()
model.fit(X, y)




client = MongoClient("mongodb+srv://anil:anil123@cluster0.dtu4quh.mongodb.net/")
db = client["email_tracker"]
collection1 = db["myapp_userstatus"]

cursor = collection1.find()
data1 = list(cursor)
data1 = pd.DataFrame(data1)
pd.set_option('display.max_columns', None)




template_links = {}

for index, row in data1.iterrows():
    email = row['email']
    template_link = row['template_link']
    name = row['name']
    
    if email in template_links:
        template_links[email].append(template_link)
    else:
        template_links[email] = [template_link]

clinet_int = data1[['_id','interests','email']]        
        
label_numbers = {
    'E-commerce': 5,
    'Technology': 12,
    'Collaboration': 2,
    'Cloud storage': 1,
    'Cloud computing': 0,
    'Social networking': 11,
    'Communication': 3,
    'Social media': 10,
    'Online marketplace': 9,
    'Messaging': 8,
    'File sharing': 6,
    'Gaming': 7,
    'Delivery': 13,
}

label_numbers2 = {
    0: "649ecbd1f558fa4952acad17",
    1: "649ecc3ef558fa4952acad19",
    2: "649ecd3cf558fa4952acad1d",
    3: "649ecd6cf558fa4952acad1f",
    5: "649ece8bf558fa4952acad23",
    6: "649eceaff558fa4952acad25",
    13: "649ed11bf558fa4952acad37",
    11: "649ed01cf558fa4952acad31",
    16: "649fb6e67c02877d2dea5423",
    12: "649ed088f558fa4952acad33",
    14: "649fb60e7c02877d2dea541f",
    15: "649fb6917c02877d2dea5421",
    17: "649fb7447c02877d2dea5425",
    7: "649ecee0f558fa4952acad27",
    8: "649ecf0df558fa4952acad29",
    9: "649ecf5cf558fa4952acad2d",
    10: "649ecfeef558fa4952acad2f",
    4: "649ece39f558fa4952acad21"
}



label_numbers = {key.lower(): value for key, value in label_numbers.items()}


clinet_int = clinet_int.reset_index(drop=True)
link_lists = {}
output = []
clinet_int


# Your previous code...

# Your previous code...

for i in range(len(clinet_int)):
    email_id = clinet_int['email'][i]  # Assuming the email IDs are stored in the 'email' column
    sentence = clinet_int['interests'][i]
    words = [word.strip() for word in sentence.split(',')]

    user_links = []  # List to store unique links for the current user

    for word in words:
        word_without_commas = word.strip()
        word_without_commas = word_without_commas.lower()  # Remove leading/trailing whitespaces
        user_interests = word_without_commas
#         print(user_interests)
        # Convert user input to label number
        label_number = label_numbers.get(user_interests)
#         print(label_number)
        if label_number is not None:
            # Predict template_id_fished for the user input
            user_pred = model.predict([[label_number]])
            user_pred = user_pred[0]  # Convert the 1D NumPy array to an integer
#             print(label_numbers2[user_pred])
            for i, template_id in enumerate(temp_df['_id']):
                if template_id == user_pred:
                    break

            # Get 5 suggestions for the same interest
            suggestions = model.predict_proba([[label_number]])
            sorted_indices = suggestions.argsort()[0][-5:][::-1]
            top_suggestions = [label_numbers2[user_pred]] + [temp_df['_id'].iloc[i] for i in sorted_indices]
#             print(top_suggestions)

            for i, template_id in enumerate(top_suggestions):
                template_name = temp_df.loc[temp_df['_id'] == template_id, 'templatePath'].iloc[0]
                if template_name not in user_links:  # Check for uniqueness before adding to the list
                    user_links.append(template_name)  # Add the link to the user's link list

        else:
            print("Interest not found in the mapping table.")

    if email_id in link_lists:
        link_lists[email_id].extend(user_links)  # Merge the user's link list with the existing list
    else:
        link_lists[email_id] = user_links

# Create the output list
for email_id in link_lists:
    output.append({
        'email': email_id,
        'links': link_lists[email_id],# No need to convert to a list as it's already a list
#         'name':name
    })
for email, templates in template_links.items():
    if email in link_lists:
        link_lists[email] = [link for link in link_lists[email] if link not in templates]

data = []

# Iterate over link_lists to extract the email and the first link for each user
for email, links in link_lists.items():
    if links:
        data.append([email, links[0]])
data

# Create a dataframe using the extracted data
df = pd.DataFrame(data, columns=['email', 'first_link'])


df

domain = {  'amazon-commande.com': ['https://templatelibrary123.s3.eu-west-3.amazonaws.com/amazon1_new.html', 'https://templatelibrary123.s3.eu-west-3.amazonaws.com/amazon2_new.html'],
    'discordfrance.com':['https://templatelibrary123.s3.eu-west-3.amazonaws.com/discord1_new.html','https://templatelibrary123.s3.eu-west-3.amazonaws.com/discord2_new.html'],
     'microsoft-login.live':['https://templatelibrary123.s3.eu-west-3.amazonaws.com/microsoft1_new.html','https://templatelibrary123.s3.eu-west-3.amazonaws.com/microsoft2_new.html',''],
     'ebay-client.com':['https://templatelibrary123.s3.eu-west-3.amazonaws.com/ebay1_new.html','https://templatelibrary123.s3.eu-west-3.amazonaws.com/ebay2_new.html','https://templatelibrary123.s3.eu-west-3.amazonaws.com/ebay4_new.html',
                        'https://templatelibrary123.s3.eu-west-3.amazonaws.com/ebay3_new.html'],
     'gmail.com':['https://templatelibrary123.s3.eu-west-3.amazonaws.com/cegid_new.html','https://templatelibrary123.s3.eu-west-3.amazonaws.com/dropbox1_new.html','https://templatelibrary123.s3.eu-west-3.amazonaws.com/dropbox2_new.html',
                  'https://templatelibrary123.s3.eu-west-3.amazonaws.com/facebook1_new.html', 'https://templatelibrary123.s3.eu-west-3.amazonaws.com/facebook2_new.html', 'https://templatelibrary123.s3.eu-west-3.amazonaws.com/github1_new.html',
                  'https://templatelibrary123.s3.eu-west-3.amazonaws.com/google1_new.html','https://templatelibrary123.s3.eu-west-3.amazonaws.com/instagram1_new.html','https://templatelibrary123.s3.eu-west-3.amazonaws.com/mailinblack_new.html',
                  'https://templatelibrary123.s3.eu-west-3.amazonaws.com/snapchat1_new.html']
 
}

for index, row in df.iterrows():
    first_link = row['first_link']
    
    # Iterate over each key-value pair in the domain dictionary
    for key, values in domain.items():
        if first_link in values:
            df.at[index, 'Domain'] = key
            break
columns_to_append = ['company_name', 'phishing_type', 'sector', 'team', 'sender_name', 'type', 'name', 'interests']

for column_name in columns_to_append:
    df[column_name] = ""
          
# df['company_name'] = data1['company']
# df['phishing_type'] = data1['phishing_type']
# df['sector'] = data1['sector']
# df['team'] = data1['team']
# df['sender_name'] = data1['sender_name']
# # df['Domain'] = 'gmail.com' 
# df['type'] = 'automated' 
# df['name'] = data1['name']
# df['interests'] = data1['interests']

for i in range(len(df['email'])):
    for j in range(len(data1['email'])):
        if df['email'][i] == data1['email'][j]:
            df['company_name'][i] = data1['company'][j]
            df['phishing_type'][i] = data1['phishing_type'][j]
            df['sector'][i] = data1['sector'][j]
            df['team'][i] = data1['team'][j]
            df['sender_name'][i] = data1['sender_name'][j]
            df['type'][i] = 'automated' 
            df['name'][i] = data1['name'][j]
            df['interests'][i] = data1['interests'][j]



df

# ---------------------------------------------------CODE PART 2---------------------------------------------------------

client = MongoClient("mongodb+srv://anil:anil123@cluster0.dtu4quh.mongodb.net/")
db = client["email_tracker"]
collection1 = db["Users"]


cursor = collection1.find()
data1 = list(cursor)

# Create DataFrame from MongoDB data
dftemp = pd.DataFrame(data1)
dftemp = dftemp.drop('_id', axis = 1)
all_users = dftemp



client = MongoClient("mongodb+srv://anil:anil123@cluster0.dtu4quh.mongodb.net/")
db = client["email_tracker"]
collection = db["myapp_userstatus"]

cursor = collection.find()
data2 = list(cursor)


# Create DataFrame from MongoDB data
dftemp = pd.DataFrame(data2)
dftemp = dftemp.drop('_id', axis = 1)
old_users = dftemp


new_users = all_users.loc[~all_users['receiver_email'].isin(old_users['email'])]
new_users

a = temp_df.loc[temp_df['difficultylevel'] == 'easy', 'templatePath'].unique().tolist()


# Assuming you have the 'new_users' DataFrame containing user information

# Initialize an empty list to store the assigned links
assigned_links = []

# Randomly assign one link from list 'a' to each user
for i in range(len(new_users)):
    assigned_link = random.choice(a)
    assigned_links.append(assigned_link)
    

# Create a new DataFrame with assigned links
assigned_users_df = pd.DataFrame({
    'name': new_users['name'],
    'company': new_users['company'],
    'sector': new_users['sector'],
    'team': new_users['team'],
    'receiver_email': new_users['receiver_email'],
    'interests': new_users['interests'],
    'assigned_link': assigned_links


})


new_simulation = pd.DataFrame(columns = ['name','email','first_link','company_name','phishing_type','sector','team','sender_name','Domain'])
new_simulation['name']=assigned_users_df['name']
new_simulation['email'] = assigned_users_df['receiver_email']
new_simulation['first_link'] = assigned_users_df['assigned_link']
new_simulation['company_name'] = assigned_users_df['company']

new_simulation['phishing_type'] = 'Email Phishing'
new_simulation['sector'] = assigned_users_df['sector']
new_simulation['team'] = assigned_users_df['team']
new_simulation['sender_name'] = 'banking075'

new_simulation['Domain'] = 'gmail.com' 
new_simulation['type'] = 'automated'
new_simulation['interests'] = assigned_users_df['interests']

for index, row in new_simulation.iterrows():
    first_link = row['first_link']
    
    # Iterate over each key-value pair in the domain dictionary
    for key, values in domain.items():
        if first_link in values:
            new_simulation.at[index, 'Domain'] = key
            break

df_f = df.append(new_simulation, ignore_index=True)

dod = ['amazon-commande.com','discordfrance.com','microsoft-login.live','ebay-client.com']

for i in range(len(df_f['Domain'])):
    if df_f['Domain'][i] in dod:
#         print(df_f['Domain'][i])
        df_f['sender_name'][i] = 'no-reply'
    else:
        df_f['sender_name'][i] = 'aais.armageddon'
#         print('no')


emails_a = all_users['receiver_email']
emails_b = df_f['email']

common_emails = set(emails_a).intersection(emails_b)
df_f = df_f[df_f['email'].isin(common_emails)]
df_f = df_f.reset_index()   
df_f = df_f.drop("index", axis = True)
# Print the updated DataFrame
# df_f = df_f[df_f['company_name'] == 'armageddon']
df_f




simulations = {}
link_counter = 1  # Counter to track simulation numbers

# Iterate through each unique first_link in the DataFrame
for link in df_f['first_link'].unique():
    # Filter the DataFrame for rows with the current first_link
    filtered_df = df_f[df_f['first_link'] == link]
    
    # Create a simulation with the current link
    simulations[link_counter] = {
        'first_link': link,
        'data': filtered_df.to_dict('records')
    }
    
    link_counter += 1  # Increment the simulation number

# Print the simulations dictionary
for simulation_number, simulation_data in simulations.items():
    print(f"Simulation {simulation_number}:")
    print(f"First Link: {simulation_data['first_link']}")
    print("Data:")
    for record in simulation_data['data']:
        print(record)
    print()
