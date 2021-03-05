if [ "$#" -lt 2 ]; then
    echo "Usage: $0 dataset_file pred_file"
    exit 1
fi
python evaluate.py $1 $2
for tag in 'in-domain' 'vocab' 'phrase' 'semantic-role' 'fault-tolerant' 'reasoning' 
do
    python evaluate.py $1 $2 --tag $tag
done

