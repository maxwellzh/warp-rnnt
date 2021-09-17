
# result=FusedBenchMarkResults.txt

# if [ -f $result ]; then
#     echo "$result file exists."
#     exit 0
# fi

mkdir -p .benchmarktmp
cp benchmark.py .benchmarktmp/
for loss in warp-rnnt-compact warp-rnnt-gather-compact warp-rnnt-fused-compact; do
    echo $loss
    CUDA_VISIBLE_DEVICES=1 python .benchmarktmp/benchmark.py \
        --loss=$loss || exit 1
    echo ""
done 
# > $result

/bin/rm -r .benchmarktmp/
