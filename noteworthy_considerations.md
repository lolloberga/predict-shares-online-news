## Dataset inspection

### General considerations
- The dataset has been acquired 06/01/2015
- Average_token_length is the average length of a word in terms of characters.
- The LDA are clusters based on a topic.
- Global_subjectivity is a number between 0 and 1.
- Global_sentiment_polarity index for positive/neutral/negative sentiment of the text.
- Title_subjectivity is a number between 0 and 1. 
- Abs_title_subjectivity is a number between 0 and 0.5


### Features descriptions
- URL: url of the article. It is not a useful feature for the predicting model because it is a sort of unique identifier of the article. In addition, the text of article can be retreived in order to use it in an words embedding but this implies the using of a pre-trained model. 
- Timedelta (numerical, ordinal, continuous in days): The number of days between the article publication and the dataset acquisition. This feature could be an interesting popularity (virality) indicator i.e. if timedelta is low and the number of corresponding shares is high, this means that the article became popular (viral) in a few days; this artifact is not always true because an article can take more days to become popular (viral).
- n_tokens_title (numerical, nominal, discrete in words): The number of words in the article's title. An initial assumption could be that a shorter title has more impact on the interest of the medium individual. From a preliminary comparison between the original title and this feature, the number does not match; it can be supposed that a stop-word elimination has been done in order to obtain the final counting. Nevertheless, some other mismatch may lead to additional processing before the count (e.g. a title inside the title). 
- n_tokens_content (numerical, nominal, discrete in words): The number of words in the article's content. It can be observed that there are some values equal to 0 words: this can be due to an error during the data collection step. A possible solution to overcome this error could be discretizing the articles based on this feature and then set the mean of the "low" bin as the final (filling) values of these articles. Another possible solution could be to get the HTML page content of these articles and set the actual number of words. An initial assumption could be the same as n_tokens_title. In addition, it is possible to observe that the number of words mismatch with respect to actual original content statistic. As for the previous feature, it can be supposed that a stop-word elimination step has been done before the count. 
- n_unique_tokens (numerical, ratio, continuous): The rate of unique words in the content. It is evaluated as: ${#unique\_words \over n\_tokens\_content}$. It is equal to 0 when $n\_tokens\_content$ is 0. 
- n_non_stop_words (numerical, ratio, continuous): The rate of non-stop-words in the content. It is evaluated as: ${#non\_stop\_words \over n\_tokens\_content}$. The distribution of this feature is misleading because it is equal to 0 when $n\_tokens\_content$ is 0 and the rest of the values is around 100%. 
- n_non_stop_unique_tokens (numerical, ratio, continuous): The rate of unique non-stop words in the content. It is evaluated as: ${#unique\_non\_stop\_words \over n\_tokens\_content}$. The variable is better distributed than the previous one and it is equal to 0 when $n\_tokens\_content$ is 0. 
- num_hrefs (numerical, ordinal, discrete): The number of external links. External means that link points out to another web-page which is not inside the Mashable domain. It is equal to 0 when $n\_tokens\_content$ is 0. The contrary does not hold because it is admissible that a news article does not have any external link. From a first comparison with original article it is possible to notice that there are some outliers in the data due to a possible error during data collection step. (e.g. there are 0 external links even though there is 1 or there are around 30 external links even though the record reports 304 external links)
- num_self_hrefs (numerical, ordinal, discrete): Number of links to other articles published by Mashable. It is equal to 0 when $n\_tokens\_content$ is 0. The contrary does not hold because it is admissible that a news article does not have any internal link. From a first comparison with original article it is possible to notice that there are some outliers in the data due to a possible error during data collection step (e.g. there are 0 internal links even though there are 2 or there are around 6 external links even though the record reports 116 internal links). 
- num_imgs (numerical, ordinal, discrete): Number of images in the article. These features admits NaN values. Even though the value is NaN, the article may actually contain some images. A possible solution to overcome this problem is to set the NaN values to 1 because from an manual inspection, the article has 1 image, another reason could be searched in the fact that the articles wher the reported number of images is 0, actually have 1 image that is the initial one. From a first comparison with original articles, it is possible to notice that there are inconsistent values of this feature, indeed some articles have reported only 1 image even though in the article there are 7 images. 




