# import library
import os
import shutil
import pandas as pd
import numpy as np
from k_means_constrained import KMeansConstrained

def create_folder(folder_path):
    """Create a folder if it doesn't exist."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")

def delete_folder(folder_path):
    """Delete a folder if it exists."""
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' deleted successfully.")
    else:
        print(f"Folder '{folder_path}' does not exist.")

def process_dataframe(data):

    # Specify the path for the new folder
    folder_path = 'result'

    # Delete the folder
    delete_folder(folder_path)

    # Create the folder
    create_folder(folder_path)

    data['id'] = range(1, len(data) + 1)
    data.rename(columns={'Lat and Long (Longitude)': 'lng'}, inplace=True)
    data.rename(columns={'Lat and Long (Latitude)': 'lat'}, inplace=True)
    data.rename(columns={'Account Name': 'name'}, inplace=True)

    # Step 1: Cluster by Rep #######################################################
    # create kmeans model/object
    clf = KMeansConstrained(
    n_clusters=6,
    size_min=270,
    size_max=560,
    random_state=0
    )

    # get features for kmeans model
    features = data[['lat', 'lng']]
    X = np.array(features)

    # do clustering
    clf.fit_predict(X)

    # save results
    labels = clf.labels_

    # send back into dataframe and display it
    data['Rep'] = labels

    # display the number of member each clustering
    _reps = data.groupby('Rep')['id'].count()
    print(_reps)

    # Number of clusters
    num_clusters = data['Rep'].nunique()
    print(f"Number of Rep:", num_clusters)
    # End Step 1 ###################################################################

    # Step 2: Cluster by Day #######################################################
    # create kmeans model/object for day
    clf_day = KMeansConstrained(
    n_clusters=5,
    size_min=35,
    size_max=65,
    random_state=0
    )

    rep_dataframes = {}

    for rep_id, rep_df in data.groupby('Rep'):
        # rep_id is the value of 'Rep' for the current group
        # rep_df is the DataFrame containing data for the current 'Rep'
        
        # Store the current 'Rep' DataFrame in the dictionary
        rep_dataframes[rep_id] = rep_df

        #  data[['lat', 'lng']]
        rep_features = rep_dataframes[rep_id][['lat', 'lng']]

        X = np.array(rep_features)

        # do clustering
        clf_day.fit_predict(X)

        # save results
        labels = clf_day.labels_

        # send back into dataframe and display it
        rep_dataframes[rep_id]['Day'] = labels
        print(f'Cluster each day by Rep ', rep_id)
        
        # save as csv file
        rep_dataframes[rep_id].to_csv(f'{folder_path}/rep_{rep_id}.csv', index=False)
        print(f'rep_{rep_id}.csv saved successfully' )

        # display the number of member each clustering
        _days = rep_dataframes[rep_id].groupby('Day')['id'].count()
        print(_days)

    # Reassign Day by sorting
    for i in range(6):
        
        print(f'Processing rep_{i}.csv')
        
        # convert csv into df
        df = pd.read_csv(f'{folder_path}/rep_{i}.csv')
        
        _days = df.groupby('Day')['id'].count().reset_index(name='count')
        print(_days)
        
        # Sort the DataFrame by 'id' in descending order
        sorted_df = _days.sort_values(by='count', ascending=False)
        print(sorted_df)
        
        # Create a dictionary where the original indices become the keys and the sorted indices become the values
        index_mapping = {original_index: sorted_index for sorted_index, original_index in enumerate(sorted_df.index)}
        index_mapping_day = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday'}

        # initialize new column
        df['NewDay'] = None
        df['NewDayName'] = None

        # update new column
        df['NewDay'] = df['Day'].map(index_mapping) 
        df['NewDayName'] = df['NewDay'].map(index_mapping_day)
        
        _days_name = df.groupby('NewDayName')['id'].count().reset_index(name='count')
        sorted_df_day = _days_name.sort_values(by='count', ascending=False)
        print(sorted_df_day)
        
        # save df to csv
        df.to_csv(f'{folder_path}/rep_{i}_updated.csv', index=False)
        print(f'rep_{i}_updated.csv saved successfully' )
    # End Step 2 ###################################################################

    # Step 3: Cluster by Week ######################################################
    # create kmeans model/object
    clf = KMeansConstrained(
    n_clusters=4,
    size_min=5,
    size_max=15,
    random_state=0
    )

    # Cluster week inside day
    for i in range(6):

        # convert csv into df
        data = pd.read_csv(f'{folder_path}/rep_{i}_updated.csv')

        day_df = {}

        for day_id, df in data.groupby('NewDay'):
            
            print(f'Processing-rep_{i}-day_{day_id}--------------')

            day_df[day_id] = df

            day_df_one_call = day_df[day_id][day_df[day_id]['F2F Calls / Month']==1]
            day_df_two_call = day_df[day_id][day_df[day_id]['F2F Calls / Month']==2]
            day_df_four_call = day_df[day_id][day_df[day_id]['F2F Calls / Month']==4]

            # get features for kmeans model
            features = day_df_one_call[['lat', 'lng']]

            X = np.array(features)

            # do clustering
            clf.fit_predict(X)

            # save results
            labels = clf.labels_

            # send back into dataframe and display it
            day_df_one_call['Week'] = labels

            # Fill up 'Week1','Week2','Week3','Week4' for one call
            day_df_one_call[['Week1','Week2','Week3','Week4']] = 0
            day_df_one_call.loc[day_df_one_call['Week'] == 0, 'Week1'] = 1
            day_df_one_call.loc[day_df_one_call['Week'] == 1, 'Week2'] = 1
            day_df_one_call.loc[day_df_one_call['Week'] == 2, 'Week3'] = 1
            day_df_one_call.loc[day_df_one_call['Week'] == 3, 'Week4'] = 1
            day_df_one_call.drop(columns=['Week'], inplace=True)

            # Fill up 'Week1','Week2','Week3','Week4' for four call
            day_df_four_call[['Week1','Week2','Week3','Week4']] = 1

            # Merge dataframe one call and four call
            day_df_one_four_call = pd.concat([day_df_one_call, day_df_four_call], ignore_index=False)

            # Fill up 'Week1','Week2','Week3','Week4' for two call
            day_df_two_call[['Week1','Week2','Week3','Week4']] = 0
            
            for index, row in day_df_two_call.iterrows():
                if index % 2 == 0:
                    day_df_two_call.loc[index, ['Week1', 'Week3']] = 1
                else:
                    day_df_two_call.loc[index, ['Week2', 'Week4']] = 1    

            # Merge dataframe one call and four call and two call
            day_df_one_two_four_call = pd.concat([day_df_one_four_call, day_df_two_call], ignore_index=False)

            # Specify the columns to concatenate and create a new column "weekid"
            columns_to_concat = ["Week1", "Week2", "Week3", "Week4"]
            day_df_one_two_four_call['weekid'] = day_df_one_two_four_call[columns_to_concat].apply(lambda row: ''.join(map(str, row)), axis=1)

            # Convert the binary values in the "weekid" column to decimal
            day_df_one_two_four_call['dec_weekid'] = day_df_one_two_four_call['weekid'].apply(lambda x: int(x, 2))
            day_df_one_two_four_call.drop(columns=['weekid'], inplace=True)
            day_df_one_two_four_call.rename(columns={'dec_weekid ': 'weekid'}, inplace=True)

            # save df to csv
            day_df_one_two_four_call.to_csv(f'{folder_path}/rep_{i}_day_{day_id}_completed.csv', index=False)
            print(f'rep_{i}_day_{day_id}_completed.csv saved successfully' )
    # End Step 3 ###################################################################

    # Step 4: Merge all rep_{i}_day_{day_id}_completed.csv #########################
    # Directory containing CSV files
    directory = folder_path

    # List all files in the directory
    files = os.listdir(directory)

    # Filter files with names ending in '_completed'
    completed_files = [file for file in files if file.endswith('_completed.csv')]

    # Check if there are any matching files
    if not completed_files:
        print("No files with names ending in '_completed' found.")
    else:
        # Read and merge the CSV files
        dfs = [pd.read_csv(os.path.join(directory, file)) for file in completed_files]
        merged_df = pd.concat(dfs, ignore_index=False)  # or use pd.merge if merging by a specific column
        
    merged_df.to_csv(f'result.csv', index=False)     

    # Step 5: Divide into seperate Rep_{rep}_{day}_Week{week}.csv #####################
    # read and convert csv into df
    data = pd.read_csv("result.csv")

    # Assuming data is your DataFrame
    for rep in range(6):
        for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
            for week in range(1, 5):
                # Filter the data for the current Rep, Day, and Week
                filtered_data = data[(data['Rep'] == rep) & (data['NewDayName'] == day)]
                
                # Sum the corresponding week column
                result_df = filtered_data[filtered_data[f'Week{week}'] == 1]
                
                # Save the DataFrame to a CSV file
                result_df.to_csv(f'{folder_path}/Rep_{rep}_{day}_Week{week}.csv', index=False)

    return "result.csv"