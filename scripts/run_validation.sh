paths=$(find  ./exported_models/*/1 -maxdepth 0)
batch_size=1

for path in ${paths[@]}; do
  echo $path
  python ./validation.py --model_path $path
done
