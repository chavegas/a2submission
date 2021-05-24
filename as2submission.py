from pyspark.sql import SparkSession, Row
from pyspark.ml.feature import PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, Normalizer, VectorAssembler, StringIndexer, IndexToString
from pyspark.ml.linalg import *
from pyspark.sql.types import * 
from pyspark.sql.functions import *
import pyspark.sql.functions as f
from pyspark.ml.recommendation import ALS

def cos_sim(a,b):
        cos_output = float(a.dot(b) / (a.norm(2) * b.norm(2)))
        return cos_output

def blank_as_null(x):
        return when(col(x) != "", col(x)).otherwise(None)

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("Assignment 2 WL 1") \
        .getOrCreate()
    spark.conf.set("spark.sql.shuffle.partitions",100)    
    tweeets_data = spark.read.option('multiline','true').json('tweets.json')    
    tweets_agg = tweeets_data.groupby("user_id").agg(f.concat_ws(" ", f.collect_list(tweeets_data.retweet_id)).alias('agg_retweets'),
                                        f.concat_ws(" ", f.collect_list(tweeets_data.replyto_id)).alias('agg_replies')).cache()
    tweets_agg = tweets_agg.withColumn("agg_retweets", blank_as_null("agg_retweets"))\
        .withColumn("agg_replies", blank_as_null("agg_replies"))  
                                    

    tweets_processed = tweets_agg.select('*',concat_ws(' ','agg_retweets','agg_replies').alias('agg_tweet_respond'))

    tokenizer = Tokenizer(inputCol='agg_tweet_respond',
        outputCol="vectors")
    tweets_vectors = tokenizer.transform(tweets_processed).cache()
    hashingTF = HashingTF(inputCol="vectors", outputCol="tf")
    tf = hashingTF.transform(tweets_vectors)

    selected_id = 202170318
    tweets_user_filtered = tf.where(f'user_id = {selected_id}')
    compare_vector = tweets_user_filtered.first()['tf']

    cos_function = udf(lambda x: cos_sim(x,compare_vector), FloatType())

    tf = tf.withColumn("CosineSim",cos_function('tf'))
    tf = tf.where(f'user_id <> {selected_id}')
    sorted_output_tf = tf.filter(tf.CosineSim > 0).sort(col('CosineSim').desc())
    sorted_output_tf.select(col('user_id').alias(f'Top Most Similar User IDs to {selected_id}'),
        col('CosineSim').alias(f'CosineSim using TFIDF Method')).show(5,truncate=False)

    from pyspark.ml.feature import CountVectorizer

    cv = CountVectorizer(inputCol="vectors", outputCol="features")

    tweets_cv = cv.fit(tweets_vectors)

    cv_data = tweets_cv.transform(tweets_vectors)
    tweets_user_filtered = cv_data.where(f'user_id = {selected_id}')
    compare_vector = tweets_user_filtered.first()['features']
    cos_function = udf(lambda x: cos_sim(x,compare_vector), FloatType())

    cv_data = cv_data.withColumn("CosineSim",cos_function('features'))
    cv_data = cv_data.where(f'user_id <> {selected_id}')
    sorted_output_cv = cv_data.filter(cv_data.CosineSim > 0).sort(col('CosineSim').desc())
    sorted_output_cv.select(col('user_id').alias(f'Top Most Similar User IDs to {selected_id}'),
        col('CosineSim').alias(f'CosineSim using CountVectorizer Method')).show(5,truncate=False)

    wl2 = tweeets_data.withColumn("mentioned_users", tweeets_data["user_mentions"].getField('id')).cache()
    wl2_users = wl2.select(col('user_id'),col('mentioned_users'))
    wl2_users = wl2_users.withColumn("mentioned_users", explode("mentioned_users"))
    wl2_users_agg = wl2_users.groupBy(col('user_id'),col('mentioned_users')).count().cache()

    stringIndexer_uid = StringIndexer(inputCol="user_id", outputCol="user_id_indexed",stringOrderType="frequencyDesc")
    model_uid = stringIndexer_uid.fit(wl2_users_agg)

    stringIndexer_mentioned = StringIndexer(inputCol="mentioned_users", outputCol="mentioned_users_indexed",stringOrderType="frequencyDesc")
    model_mu = stringIndexer_mentioned.fit(wl2_users_agg)

    td = model_uid.transform(wl2_users_agg)
    wl2_users_transformed = model_mu.transform(td)

    als = ALS(userCol="user_id_indexed", itemCol="mentioned_users_indexed", ratingCol="count",
            coldStartStrategy="drop")
    model = als.fit(wl2_users_transformed)

    model_recs = model.recommendForAllUsers(5).cache()

    uid_labels = model_uid.labels
    uid_labels_ = array(*[lit(x) for x in uid_labels])

    n = 5
    mu_labels = model_mu.labels
    mu_labels_ = array(*[lit(x) for x in mu_labels])
    recommendations = array(*[struct(
        mu_labels_[col("recommendations")[i]["mentioned_users_indexed"]].alias("userId"),
        col("recommendations")[i]["rating"].alias("rating")
    ) for i in range(n)])

    model_recs = model_recs.withColumn("recommendations", recommendations)\
            .withColumn("user_id", uid_labels_[col("user_id_indexed")])
    model_recs.select(['user_id'] + [col("recommendations")[i].alias('Recommended User: '+str(i+1)) for i in range(n)]).show(truncate=False)            