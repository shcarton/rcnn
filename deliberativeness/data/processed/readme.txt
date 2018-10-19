A "processed" data file should have two things: a CSV and one or more metadata files

The CSV should have the following columns:

text: the processed (tokenized, etc) text of the comment
datetime: a datetime string
platform_id: ID of the platform
platform_comment_id: platform-specific comment ID
url: URL for this particular comment
original_text: the unprocessed text of the comment
tokenization: a description of how the unprocessed text is tokenized into the processed text
target: if there is a label column, it should be called target



Every data file should be accompanied with a platform.json file which describes the following attributes of the dataset:

platform_id: an arbitrarily chosen ID for each platform
display_title: a short title for the platform (e.g. Wikipedia.org)
url: URL for the platform
size: sample size for the dataset representing this platform
start_date: datetime string representing the date of the earliest comment in this dataset
end_date: datetime string representing the date of the latest comment for this dataset
description: a description of the dataset


If the datafile has a target column, there should also be a dimension.json file that describes the following:
dimension_id: an arbitrarily-chosen dimension ID
name: name of the dimension
description: description
source_url: URL (if available) for the dataset
platform_id: ID of the platform on which the dimension is defined
irr: inter-rater reliability (when available)
size: number of samples
