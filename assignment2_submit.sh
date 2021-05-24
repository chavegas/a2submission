spark-submit \
    --master yarn \
    --deploy-mode cluster \
    --num-executors 3 \
    as2submission.py \
    --output $1 
