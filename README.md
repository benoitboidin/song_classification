# Audio Genre Classification Challenge

Welcome to the Audio Genre Classification Challenge repository! This repository contains datasets and information for a challenge focused on audio genre classification. Below is an overview of the provided data and files:

## Data Description

### Audio Files
- **train.zip**: Archive containing audio files in MP3 format for the train dataset. Approximately 4000 tracks are included in this set, totaling around 3.6 GB.
- **test.zip**: Archive containing audio files in MP3 format for the test dataset. Similar to the train set, it also contains around 4000 tracks, totaling approximately 3.6 GB. Genre information is not provided for this set.

### Metadata Files
- **train.csv**: CSV file indicating the genre (genre_id) of each track in the train set.
- **test.csv**: CSV file providing track_ids for the tracks in the test set, for which genre estimation is required.
- **genres.csv**: CSV file listing the possible genres along with their corresponding ids. Eight genres are considered in this challenge.

### Sample Submission
- **sampleSubmission.csv**: An example submission file that can be used for evaluation. It has been randomly generated and can be evaluated based on category accuracy.

## Example
As an example, let's consider the file `000002.mp3`, which is an audio file from the train set. Its genre is indicated in `train.csv`. Genre_id 1 is associated with the track_id `000002`. Referring to `genres.csv`, it is indicated that genre_id 1 corresponds to the Classical genre.

## Objective
The objective of this challenge is to predict the genre for the audio files in the test set based on the provided training data.

Feel free to explore the data and participate in the challenge! If you have any questions or need further clarification, don't hesitate to reach out. Good luck!
