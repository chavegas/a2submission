#Load in required functions
from pyspark.sql import SparkSession, Row
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, Normalizer, VectorAssembler, StringIndexer, IndexToString, CountVectorizer
from pyspark.ml.linalg import *
from pyspark.sql.types import * 
from pyspark.sql.functions import *
import pyspark.sql.functions as f
from pyspark.ml.recommendation import ALS

#Function to calculate cosine similarity
def cos_sim(a,b):
        cos_output = float(a.dot(b) / (a.norm(2) * b.norm(2)))
        return cos_output

#Function to set blank string columns to null
def blank_as_null(x):
        return when(col(x) != "", col(x)).otherwise(None)

#Function to process the data for WL1
def process_data_wl1(data):
    #Group data by user_id and concatenate each entry of replies and retweets
    tweets_agg = data.groupby("user_id").agg(f.concat_ws(" ", f.collect_list(data.retweet_id)).alias('agg_retweets'),
                                        f.concat_ws(" ", f.collect_list(data.replyto_id)).alias('agg_replies')).cache()
    #Set null columns to blank                                    
    tweets_agg = tweets_agg.withColumn("agg_retweets", blank_as_null("agg_retweets"))\
        .withColumn("agg_replies", blank_as_null("agg_replies"))  
    #Combine retweets and replies columns    
    tweets_processed = tweets_agg.select('*',concat_ws(' ','agg_retweets','agg_replies').alias('agg_tweet_respond'))
    #Unpersist to free memory
    tweets_agg.unpersist()
    #Tokenize the data for processing
    tokenizer = Tokenizer(inputCol='agg_tweet_respond',
        outputCol="vectors")
    tweets_vectors = tokenizer.transform(tweets_processed)
    return tweets_vectors

#Function to apply workload 1, if statement to determing the word transform
def apply_tf1(data,hashtf):
    if hashtf == True:
        hashingTF = HashingTF(inputCol="vectors", outputCol="tf")
        tf = hashingTF.transform(data).cache()
    else:
        cv = CountVectorizer(inputCol="vectors", outputCol="tf")
        tweets_cv = cv.fit(data)
        tf = tweets_cv.transform(data).cache()

    #Selected ID to compare users to. Take this ID, filter to data and extract the transformed feature of this ID
    selected_id = 202170318
    tweets_user_filtered = tf.where(f'user_id = {selected_id}')
    compare_vector = tweets_user_filtered.first()['tf']
    #Define UDF of cosine function
    cos_function = udf(lambda x: cos_sim(x,compare_vector), FloatType())
    #Apply cosine function and remove the user ID row to prevent 1:1 match
    tf = tf.withColumn("CosineSim",cos_function('tf'))
    tf = tf.where(f'user_id <> {selected_id}')
    #Sort by top users
    sorted_output_tf = tf.filter(tf.CosineSim > 0).sort(col('CosineSim').desc())
    tf.unpersist()
    return sorted_output_tf.select(col('user_id').alias(f'Top Most Similar User IDs to {selected_id}'),
        col('CosineSim').alias(f'CosineSim')).show(5,truncate=False)

#Function to process the data for WL2
def process_data_wl2(data):
    #Extract the id field from the user_mentions column
    wl2 = data.withColumn("mentioned_users", data["user_mentions"].getField('id')).cache()
    #Select only key columns
    wl2_users = wl2.select(col('user_id'),col('mentioned_users'))
    #Explode row-wise each mentioned user in the mentioned_users column from nested array to individual rows
    wl2_users = wl2_users.withColumn("mentioned_users", explode("mentioned_users"))
    #Aggregate each mentioned user by count
    wl2_users_agg = wl2_users.groupBy(col('user_id'),col('mentioned_users')).count()
    wl2_users.unpersist()
    return wl2_users_agg

#Function to apply WL2
def apply_wl2(data):
    #Set string indexers for transformation in the model for each column and fit the models
    stringIndexer_uid = StringIndexer(inputCol="user_id", outputCol="user_id_indexed",stringOrderType="frequencyDesc")
    model_uid = stringIndexer_uid.fit(data)

    stringIndexer_mentioned = StringIndexer(inputCol="mentioned_users", outputCol="mentioned_users_indexed",stringOrderType="frequencyDesc")
    model_mu = stringIndexer_mentioned.fit(data)
    #Transform based on string indexers
    td = model_uid.transform(data)
    wl2_users_transformed = model_mu.transform(td)
    #apply ALS algorithm and fit to the data
    als = ALS(userCol="user_id_indexed", itemCol="mentioned_users_indexed", ratingCol="count",
            coldStartStrategy="drop")
    model = als.fit(wl2_users_transformed)

    #Build recommendations for each user and cache the output
    model_recs = model.recommendForAllUsers(5).cache()

    #Extract labels and turn them into an array to retransform from StringIndexer
    uid_labels = model_uid.labels
    uid_labels_ = array(*[lit(x) for x in uid_labels])
    n = 5
    mu_labels = model_mu.labels
    mu_labels_ = array(*[lit(x) for x in mu_labels])
    #Apply labels for each nested recommendation in the recommendations column
    recommendations = array(*[struct(
        mu_labels_[col("recommendations")[i]["mentioned_users_indexed"]].alias("userId"),
        col("recommendations")[i]["rating"].alias("rating")
    ) for i in range(n)])
    #Apply remapped labels for recommendations and map the user_ids back from StringIndexer
    model_recs = model_recs.withColumn("recommendations", recommendations)\
            .withColumn("user_id", uid_labels_[col("user_id_indexed")])
    #Split the column out and return it such that there are 5 columns for each recommended user            
    return model_recs.select(['user_id'] + [col("recommendations")[i].alias('Recommended User: '+str(i+1)) for i in range(n)]).show(truncate=False)            

#Run the functinons
if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("Assignment 2 WL 1") \
        .getOrCreate()
    spark.conf.set("spark.sql.shuffle.partitions",100)    
    tweeets_data = spark.read.option('multiline','true').json('tweets.json')    
    #Apply functions
    tweets_processed = process_data_wl1(tweeets_data).cache()

    apply_tf1(tweets_processed,True)
    apply_tf1(tweets_processed,False)

    tweets_processedwl2 = process_data_wl2(tweeets_data).cache()
    apply_wl2(tweets_processedwl2)    