paths=$(find  ./exported_models/*/1 -maxdepth 0)
batch_size=128

for path in ${paths[@]}; do
  echo $path, $batch_size
  python ./benchmark_local.py --model_path $path --batch_size $batch_size
done