import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



activity_labels = {
    1: 'walking',
    2: 'running',
    3: 'shuffling',
    4: 'stairs (ascending)',
    5: 'stairs (descending)',
    6: 'standing',
    7: 'sitting',
    8: 'lying',
    13: 'cycling (sit)',
    14: 'cycling (stand)',
    130: 'cycling (sit, inactive)',
    140: 'cycling (stand, inactive)'
}

indexed_labels = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    13: 8,
    14: 9,
    130: 10,
    140: 11
}

csvs = {
    1: "harth/S006.csv",
    2: "harth/S008.csv",
    3: "harth/S009.csv",
    4: "harth/S010.csv",
    5: "harth/S012.csv",
    6: "harth/S013.csv",

}

sensors = ["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]
#Load multiple cvs files in a dataframe 
def load_data():
    data = pd.DataFrame()
    for _, value in csvs.items():
        temp_data = pd.read_csv(value)
        data = pd.concat([data, temp_data])
    return data


def activityDistribution(df):
    activity_time = pd.DataFrame()
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds().fillna(0)
    activity_time = activity_time.fillna(activity_time.mean()) 
    activity_time = df.groupby('activity_label')['time_diff'].sum()
    #plot activity time

    activity_time.plot(kind='bar', title='Total Time Spent per Activity')

    plt.show()


def sensorValueDistribution(df):
    # Visualize sensor distributions
    plt.figure(figsize=(15, 10))
    for i, column in enumerate(df.columns[1:-2], 1):  # Skip timestamp, label, and activity_label
        plt.subplot(3, 3, i)
        sns.histplot(df[column], kde=True)
        plt.title(column)
    plt.tight_layout()
    #visualize sensor distributions violin plot
    df_std = df[sensors]
    df_std = df_std.melt(var_name='Column', value_name='values')

    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(x='Column', y='values', data=df_std)
    _ = ax.set_xticklabels(df.keys(), rotation=90)
    plt.show()

def sensorActivityCorrelation(df):
    #sensor activity heatmap
    mean_values = df.groupby('activity_label').agg({
    'back_x': 'mean',
    'back_y': 'mean',
    'back_z': 'mean',
    'thigh_x': 'mean',
    'thigh_y': 'mean',
    'thigh_z': 'mean'
    }).reset_index()
    #multiply by 100 to get percentage
    mean_values = mean_values * 100
    test = mean_values.set_index('activity_label')
    sns.heatmap(test,cmap="Blues")
    plt.savefig('activity.png')
    #sensor correlation heatmap
    val = test.corr()

    sns.heatmap(val)
    #save the heatmap
    plt.savefig('sensor_correlation.png')
    plt.plot()

print("Loading data...")
df = load_data()


print("Transforming data...")
df['activity_label'] = df['label'].map(activity_labels)

#df['timestamp'] = pd.to_datetime(df['timestamp'])

print("Ploating activity distribution...")
sensorActivityCorrelation(df)